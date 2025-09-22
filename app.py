import os, re, json, time, uuid
import requests
import gradio as gr
import logging, sys
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CONNECT_TIMEOUT = float(os.getenv("SEPSIS_API_CONNECT_TIMEOUT", "30"))
READ_TIMEOUT_DEFAULT = float(os.getenv("SEPSIS_API_READ_TIMEOUT", "120"))
READ_TIMEOUT_S1 = float(os.getenv("SEPSIS_API_READ_TIMEOUT_S1", str(READ_TIMEOUT_DEFAULT)))
READ_TIMEOUT_S2 = float(os.getenv("SEPSIS_API_READ_TIMEOUT_S2", "180"))  # S2 is heavier; default 180s

def _retry_session():
    s = requests.Session()
    retry = Retry(
        total=2,                # 2 quick retries on transient errors
        backoff_factor=0.5,     # 0.5s, 1.0s
        status_forcelist=[502, 503, 504],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = _retry_session()

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("sepsis-agent")

# ------------------------------
# State helpers
# ------------------------------

def new_state():
    # brand-new per-conversation state
    return {
        "sheet": None,
        "conv_id": None,
        "session": str(uuid.uuid4()),
        "awaiting_unvalidated_s2": False,
    }

# -------- Smart readiness + guidance helpers --------

S1_ASK_ORDER = [
    "age.months", "sex", "adm.recent", "wfaz", "cidysymp",
    "not.alert", "hr.all", "rr.all", "envhtemp", "crt.long",
]

S1_FRIENDLY = {
    "age.months":  "Age in months",
    "sex":         "Sex (1 = male, 0 = female)",
    "adm.recent":  "Overnight hospitalisation in the last 6 months? (1 = yes, 0 = no)",
    "wfaz":        "Weight-for-age Z-score",
    "cidysymp":    "Duration of illness (days)",
    "not.alert":   "Not alert (AVPU < A)? (1 = yes, 0 = no)",
    "hr.all":      "Heart rate (bpm)",
    "rr.all":      "Respiratory rate (/min)",
    "envhtemp":    "Axillary temperature (°C)",
    "crt.long":    "Capillary refill time > 2 s? (1 = yes, 0 = no)",
}

S2B_ORDER = ["oxy.ra", "CRP", "IL6", "CXCl10"]
S2B_FRIENDLY = {
    "oxy.ra":  "SpO₂ on room air (%)",
    "CRP":     "CRP (mg/L)",
    "IL6":     "IL-6 (pg/mL)",
    "CXCl10":  "CXCL10 (pg/mL)",
}

def s1_ready(sheet: dict) -> bool:
    clin = (sheet or {}).get("features", {}).get("clinical", {}) or {}
    return len(missing_for_s1(clin)) == 0

def s1_decision(sheet: dict) -> str:
    from_str = _first(((sheet or {}).get("s1") or {}).get("s1_decision"))
    return _norm_key(from_str)

def s2_ready(sheet: dict) -> bool:
    feats = (sheet or {}).get("features", {}) or {}
    clin = feats.get("clinical", {}) or {}
    labs = feats.get("labs", {}) or {}
    merged = {**clin, **labs}
    return validated_set_name(merged) is not None

def missing_for_s2_setB(sheet: dict):
    feats = (sheet or {}).get("features", {}) or {}
    clin = feats.get("clinical", {}) or {}
    labs = feats.get("labs", {}) or {}
    merged = {**clin, **labs}
    missing = []
    for k in S2B_ORDER:
        v = merged.get(k)
        if v in (None, "", 0, 0.0):
            missing.append(k)
    return missing

def build_s1_missing_prompt(missing_keys) -> str:
    # Compact “opening message” variant: only show the items still missing
    lines = [
        "This is clinical decision support, not a diagnosis.",
        "",
        "To run Stage 1 (S1), please provide the remaining details:",
    ]
    for k in missing_keys:
        lines.append(f"• {S1_FRIENDLY.get(k, k)}")
    return "\n".join(lines)

def build_s2_missing_prompt(missing_keys) -> str:
    lines = [
        "To run Stage 2 (S2) with Set B, please add:",
    ]
    for k in missing_keys:
        lines.append(f"• {S2B_FRIENDLY.get(k, k)}")
    # Optional hint about Set A:
    lines.append("Alternatively, Set A can be used: CRP, TNFR1, suPAR, and SpO₂ on room air.")
    return "\n".join(lines)

def build_guidance_after_update(sheet: dict) -> str:
    """
    Returns the best next message after the Info Sheet was updated:
      - If S1 not run yet:
          - If S1 ready → 'Info Sheet updated. If the Info Sheet looks right, press Run S1.'
          - Else → S1 missing prompt (no 'Info Sheet updated.' prefix)
      - If S1 run and decision is OTHER:
          - If Set B (or full/Set A) is ready → 'Info Sheet updated. If you’re ready, press Run S2.'
          - Else → S2 missing prompt (Set B) (no 'Info Sheet updated.' prefix)
      - Else (S1 Severe/NotSevere) → terse acknowledgement.
    """
    if "s1" not in (sheet or {}):
        if s1_ready(sheet):
            return "Info Sheet updated. If the Info Sheet looks right, press **Run S1**."
        missing = missing_for_s1((sheet.get("features", {}).get("clinical", {})) if sheet else {})
        return build_s1_missing_prompt(missing)

    # S1 exists
    dec = s1_decision(sheet)
    if dec == "OTHER":
        if s2_ready(sheet):
            return "Info Sheet updated. If you’re ready, press **Run S2**."
        missing = missing_for_s2_setB(sheet)
        return build_s2_missing_prompt(missing)

    # S1 was Severe or NotSevere; keep it minimal
    return "Info Sheet updated."

# ------------------------------
# OpenAI client & config
# ------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
USE_LLM_DEFAULT = True
PARSER_MODE = os.getenv("PARSER_MODE", "llm_only").strip().lower()
DEBUG_AGENT = bool(int(os.getenv("DEBUG_AGENT", "0")))


def _get_llm_model():
    m = os.getenv("LLM_MODEL_ID", "").strip()
    if not m or "/" in m:
        return "gpt-4o-mini"
    return m

# ------------------------------
# Agent system prompt
# Only the constraints, tool contract, and API rules live here.
# All surface dialogue is authored by the LLM (no host-scripted text).
# ------------------------------
AGENT_SYSTEM = r"""
You are Sepsis Spotter, a clinical intake and orchestration assistant (experimental research preview; not a diagnosis).

# Role
- Help the user build a correct Info Sheet JSON (current_sheet) for Stage 1 (S1) and Stage 2 (S2).
- **You do not call the model APIs.** The user will press buttons (“Run S1”, “Run S2”). Keep all dialogue natural and concise.
- Use plain language only (never show raw keys). No emojis.

# How to act
- Parse the user’s message; when you can extract fields, emit a single tool call:
  {"action":"update_sheet","features":{"clinical":{...},"labs":{...}}}
- If something critical is missing, emit exactly one focused request:
  {"action":"ask","message":"<one plain-language question for the single most important missing item>"}
- Never invent values. Convert years → months for age. Map sex: 1 = male, 0 = female.
- Be concise; don’t paste the Info Sheet JSON (the UI shows it). Don’t restate all values the user typed.
- Avoid repetition: don’t repeat “Current info sheet updated” or the “press Run S1/Run S2” line in consecutive turns unless new info was added.
- If the user asks for a summary of the current sheet, provide a brief plain-language summary of what’s present and what's missing for S1/S2.

# Buttons & consent
- When you believe the minimal S1 set is present, say:
  If the Info Sheet looks right, press **Run S1**.
- After S1 returns “Other”, suggest that the user provide additional information in the form of a validated lab set for S2; when a validated S2 set plus room-air SpO₂ is present, say:
  If you’re ready, press **Run S2** with **Set A** or **Set B** (as applicable).
- If labs are incomplete or not validated, ask for the missing pieces (one at a time) or state that the combination is not validated and let the user decide.

# S1 minimal (must exist in current_sheet to be ready)
age.months, sex (1/0), adm.recent (1/0), wfaz, cidysymp, not.alert (1/0), hr.all, rr.all, envhtemp, crt.long (1/0)

# S2 validated sets (plus SpO₂ on room air)
Set A: CRP, TNFR1, suPAR, oxy.ra
Set B: CRP, CXCL10, IL6, oxy.ra
Full-panel allowed if many labs are present (~6+). When asking for labs, prefer Set B by default (typical priority: SpO₂ on room air, CRP, IL-6, CXCL10).

# Interaction Flow
- **First user turn** AND the current_sheet has no collected fields (both features.clinical and features.labs are empty) → send this exact message, with no extra text before or after:

  This is clinical decision support, not a diagnosis.

  To run Stage 1 (S1), please share these minimal required details:
  • Age in months
  • Sex (1 = male, 0 = female)
  • Overnight hospitalisation within the last 6 months (1 = yes, 0 = no)
  • Weight for age z-score
  • Duration of illness (days)
  • Not alert? (AVPU < A) (1 = yes, 0 = no)
  • Heart rate (bpm)
  • Respiratory rate (/min)
  • Axillary temperature (°C)
  • Capillary refill time greater than 2 seconds? (1 = yes, 0 = no)

  If you have more information S1 can also use, please include any of: comorbidity, wasting, stunting, prior care, travel time ≤1h, diarrhoeal syndrome, WHO pneumonia or severe pneumonia, prostration, intractable vomiting, convulsions, lethargy, IMCI danger sign, parenteral treatment before enrolment, and SIRS score.

  Let me know if you have any questions.

- Do not repeat this block again in later turns.

# Response Guarantees (STRICT)
- **Never** reply with only “Info Sheet updated” or similar.
- After any `update_sheet` tool call, your user-visible message **must end with exactly one directive**:
  • If S1 has **not** been run and the S1 minimal set is complete → say: *If the Info Sheet looks right, press **Run S1***.
  • If S1 has been run and the decision is **Other**:
      – If Set A or Set B (with room-air SpO₂) is complete → say: *If you’re ready, press **Run S2** with Set A/B*.
      – Otherwise, issue {"action":"ask"} for **exactly one** missing item (prefer Set B defaults: SpO₂ on room air, CRP, IL-6, CXCL10).
  • If S1 has been run and the decision is **Severe** or **NOTSevere** → acknowledge briefly and do **not** provide treatment recommendations.
- Avoid repeating the same “press Run …” line in back-to-back turns unless new information was added.
- Keep messages short: a brief acknowledgement of the update, then the single directive/question.
- When you use {"action":"ask"}, you MUST also include the exact same question in your assistant message. Do not reply with generic acknowledgements.

# Result summaries
- After the host app runs S1/S2 and attaches results in context, provide a brief, plain-language recap only (no treatment “next steps”), then include a short decision-support disclaimer in your own words (e.g., “This supports clinical decision-making and is not a diagnosis.”).

# Field dictionary (key → meaning)
*NOTE: Never show these keys to the user; use plain language in chat.*

- age.months → Age in months
- sex → Sex (1=male, 0=female)
- bgcombyn → Comorbidity present (1/0)
- adm.recent → Overnight hospitalisation within the last 6 months (1/0)
- wfaz → Weight-for-age Z-score
- waste → WFL Z < −2 (1/0)
- stunt → LAZ < −2 (1/0)
- cidysymp → Duration of illness (days)
- prior.care → Prior care-seeking (1/0)
- travel.time.bin → Travel time ≤1h (1/0)
- diarrhoeal → Diarrhoeal syndrome (1/0)
- pneumo → WHO pneumonia (1/0)
- sev.pneumo → WHO severe pneumonia (1/0)
- ensapro → Prostration/encephalopathy (1/0)
- vomit.all → Intractable vomiting (1/0)
- seiz → Convulsions (1/0)
- pfacleth → Lethargy (1/0)
- not.alert → Not alert (AVPU < A) (1/0)
- danger.sign → Any IMCI danger sign (1/0)
- hr.all → Heart rate (bpm)
- rr.all → Respiratory rate (/min)
- oxy.ra → SpO₂ on room air (%)
- envhtemp → Axillary temperature (°C)
- crt.long → Capillary refill >2 s (1/0)
- parenteral_screen → Parenteral treatment before enrolment (1/0)
- SIRS_num → SIRS score (0–4)
"""

# ------------------------------
# Tool schema exposed to the LLM
# ------------------------------
TOOL_SPEC = [{
    "type": "function",
    "name": "sepsis_command",
    "description": "Update the current Info Sheet from user-provided data or ask for exactly one missing item.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["ask", "update_sheet"]},
            "message": {"type": "string"},
            "features": {
                "type": "object",
                "properties": {
                    "clinical": {"type": "object", "additionalProperties": True},
                    "labs": {"type": "object", "additionalProperties": True},
                }
            }
        },
        "required": ["action"],
        "additionalProperties": False
    }
}]

# ------------------------------
# API endpoints & field lists
# ------------------------------
API_S1 = os.getenv("SEPSIS_API_URL_S1", "https://sepsis-spotter-beta.onrender.com/s1_infer")
API_S2 = os.getenv("SEPSIS_API_URL_S2", "https://sepsis-spotter-beta.onrender.com/s2_infer")

S1_REQUIRED_MIN = [
    "age.months", "sex", "adm.recent", "wfaz", "cidysymp", "not.alert",
    "hr.all", "rr.all", "envhtemp", "crt.long"
]

LAB_KEYS = [
    "CRP", "TNFR1", "supar", "CXCl10", "IL6", "IL10", "IL1ra", "IL8", "PROC",
    "ANG1", "ANG2", "CHI3L", "STREM1", "VEGFR1", "lblac", "lbglu", "enescbchb1"
]

LAB_CANON = {
    "crp": "CRP",
    "tnfr1": "TNFR1",
    "supar": "supar",
    "cxcl10": "CXCl10",   # ← canonical uses lowercase 'l'
    "il6": "IL6",
    "il10": "IL10",
    "il1ra": "IL1ra",
    "il8": "IL8",
    "proc": "PROC",
    "ang1": "ANG1",
    "ang2": "ANG2",
    "chi3l": "CHI3L",
    "strem1": "STREM1",
    "vegfr1": "VEGFR1",
    "lblac": "lblac",
    "lbglu": "lbglu",
    "enescbchb1": "enescbchb1",
    # SpO2 synonyms (if the LLM ever writes a "lab" key for it)
    "spo2": "oxy.ra",
    "spo₂": "oxy.ra",
    "oxygen": "oxy.ra",
    "sat": "oxy.ra",
}

def _normkey(k: str) -> str:
    # lowercase + strip non-alphanum so "CXCL-10" and "cxcl10" collapse
    return re.sub(r"[^a-z0-9]+", "", (k or "").lower())

def canonicalize_features(feats: dict) -> dict:
    feats = feats or {}
    clin_in = (feats.get("clinical") or {}).copy()
    labs_in = (feats.get("labs") or {}).copy()

    clin_out = dict(clin_in)
    labs_out = {}

    for k, v in labs_in.items():
        nk = _normkey(k)
        canon = LAB_CANON.get(nk)
        if canon == "oxy.ra":
            # make sure SpO2 lands in clinical
            clin_out["oxy.ra"] = v
        elif canon:
            labs_out[canon] = v
        else:
            labs_out[k] = v  # unknown key: keep as-is

    return {"clinical": clin_out, "labs": labs_out}

S_DISCLAIMER = (
    "This is clinical decision support, not a diagnosis. You must use your own clinical judgment, "
    "training, and knowledge to make referral or treatment decisions. No liability is accepted."
)

def _first(x):
    return x[0] if isinstance(x, (list, tuple)) and x else x

def _norm_key(s):
    return re.sub(r"\s+|\.", "", str(s or "")).upper()

def format_s1_output(s1_json: dict) -> str:
    decision = _first((s1_json or {}).get("s1_decision"))
    key = _norm_key(decision)
    if key == "SEVERE":
        body = (
            "Model decision: SEVERE. According to historical data and model specifics, the given patient’s symptoms "
            "suggest a severe disease. That is, the child either died within two days of enrolment, required organ "
            "support such as mechanical ventilation, inotropic therapy, or renal replacement within two days, or was "
            "discharged home to die during this period."
        )
    elif key in ("NOTSEVERE", "NOTSEVERE"):
        body = (
            "Model decision: NOT SEVERE. According to historical data and model specifics, the given patient’s symptoms "
            "suggest a non-severe disease. That is, the child was not admitted to any health facility and all symptoms "
            "had resolved by day 28."
        )
    else:
        body = (
            "S1 decision: OTHER. According to model specifics, laboratory tests/biomarkers are required to make a more "
            "informed outcome prediction."
            "If you have any laboratory results available, please provide them to run Stage S2 (S2)."
            "If you want to know which laboratory tests are compatible OR which minimal sets may be used, please ask."
        )
    return f"{body}\n\n{S_DISCLAIMER}"

def _extract_s2_call(s2_json) -> str:
    try:
        if isinstance(s2_json, list) and s2_json:
            return str(s2_json[0].get("call") or "")
    except Exception:
        pass
    try:
        return str(s2_json.get("call") or "")
    except Exception:
        return ""

def format_s2_output(s2_json: dict) -> str:
    decision = _extract_s2_call(s2_json)
    key = _norm_key(decision)

    if key == "SEVERE":
        body = (
            "Model decision: SEVERE. According to historical data and model specifics, the given patient’s symptoms "
            "suggest a severe disease. That is, the child either died within two days of enrolment, required organ "
            "support such as mechanical ventilation, inotropic therapy, or renal replacement within two days, or was "
            "discharged home to die during this period."
        )
    elif key in ("NOTSEVERE", "NOTSEVERE"):
        body = (
            "Model decision: NOT SEVERE. According to historical data and model specifics, the given patient’s symptoms "
            "suggest a non-severe disease. That is, the child was not admitted to any health facility and all symptoms "
            "had resolved by day 28."
        )
    elif key in ("PROBSEVERE", "PROBABLESEVERE"):
        body = (
            "S2 decision: PROBABLE SEVERE\n"
            "According to historical data and model specifics, the given patient’s symptoms suggest a probable severe disease. "
            "That is, the child died after the first two days and before day 28 without meeting criteria for severe disease, "
            "or required more than two days of hospital admission before day 28 without meeting criteria for severe disease."
        )
    elif key in ("PROBNONSEVERE", "PROBABLENONSEVERE"):
        body = (
            "S2 decision: PROBABLE NON-SEVERE\n"
            "According to historical data and model specifics, the given patient’s symptoms suggest a probable non-severe disease. "
            "That is, the child was admitted for two days or less before day 28 without criteria for severe or probable severe disease, "
            "or was not admitted to hospital but still had ongoing symptoms at day 28."
        )
    else:
        body = f"S2 decision: {decision or 'Unavailable'}."

    return f"{body}\n\n{S_DISCLAIMER}"

# ------------------------------
# Info sheet helpers
# ------------------------------

def new_sheet(clinical=None, labs=None):
    return {
        "sheet_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "patient": {"anon_id": f"anon-{int(time.time())}"},
        "features": {"clinical": clinical or {}, "labs": labs or {}},
        "notes": []
    }


def merge_sheet(sheet, add_clin, add_labs):
    out = dict(sheet)
    out.setdefault("features", {}).setdefault("clinical", {}).update(add_clin or {})
    out.setdefault("features", {}).setdefault("labs", {}).update(add_labs or {})
    return out


def merge_features(sheet, feats):
    feats = canonicalize_features(feats)  # ← normalize first
    return merge_sheet(
        sheet,
        feats.get("clinical") or {},
        feats.get("labs") or {},
    )

def s1_min_ready(sheet: dict) -> bool:
    clin = (sheet or {}).get("features", {}).get("clinical", {}) or {}
    return len(missing_for_s1(clin)) == 0

def s2_enabled(sheet: dict) -> bool:
    # Keep S2 disabled until S1 has been run at least once
    return bool((sheet or {}).get("s1"))

def compute_btn_states(st: dict):
    sheet = (st or {}).get("sheet") or {}
    return (
        gr.update(interactive=s1_min_ready(sheet)),     # for btn_s1
        gr.update(interactive=s2_enabled(sheet)),       # for btn_s2
    )

# ------------------------------
# Lightweight legacy extractor (fallback when LLM unavailable)
# ------------------------------

def extract_features(text: str):
    clinical, labs = {}, {}
    t = (text or "").strip()

    # Age
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)\b", t, re.I)
    if m: clinical["age.months"] = float(m.group(1)) * 12
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?|mo)\b", t, re.I)
    if m: clinical["age.months"] = float(m.group(1))

    # Sex
    if re.search(r"\bmale\b|\bboy\b", t, re.I): clinical["sex"] = 1
    if re.search(r"\bfemale\b|\bgirl\b", t, re.I): clinical["sex"] = 0

    # Admission last 6 months
    if re.search(r"(overnight|over\s*night)\s+(hospitali[sz]ation|admission).*(last|past)\s*(six|6)\s*months", t, re.I):
        clinical["adm.recent"] = 1
    if re.search(r"\bno\b.*(hospitali[sz]ation|admission).*(last|past)\s*(six|6)\s*months", t, re.I):
        clinical["adm.recent"] = 0

    # WFA z
    m = re.search(r"weight\s*for\s*age\s*z\s*-?\s*score\s*(?:is|:)?\s*(-?\d+(?:\.\d+)?)", t, re.I)
    if not m:
        m = re.search(r"\bwfaz\b\s*[:=]\s*(-?\d+(?:\.\d+)?)", t, re.I)
    if m: clinical["wfaz"] = float(m.group(1))

    # Duration days
    m = re.search(r"(duration of (?:illness|symptoms?)|illness duration)\s*(?:is|:)?\s*(\d+(?:\.\d+)?)\s*(days?|d)\b", t, re.I)
    if m: clinical["cidysymp"] = int(float(m.group(1)))

    # Alertness
    if re.search(r"\bnot alert\b|\bAVPU\b.*<\s*A", t, re.I):
        clinical["not.alert"] = 1

    # CRT
    if re.search(r"cap(illary)?\s*refill.*(>\s*2|greater than\s*2)\s*s", t, re.I):
        clinical["crt.long"] = 1
    elif re.search(r"cap(illary)?\s*refill.*(≤\s*2|<\s*2|within\s*2\s*s|normal)", t, re.I):
        clinical["crt.long"] = 0

    # Temp
    m = re.search(r"(axillary\s+temperature|temperature)\s*(in\s*celsius)?\s*(is|:)?\s*([0-9]{2}(?:\.[0-9]+)?)", t, re.I)
    if m: clinical["envhtemp"] = float(m.group(4))

    # HR / RR
    m = re.search(r"\bHR[:\s]*([0-9]{2,3})\b", t, re.I) or re.search(r"heart\s*rate\s*(is|:)?\s*([0-9]{2,3})\s*bpm", t, re.I)
    if m: clinical["hr.all"] = int(m.group(m.lastindex))
    m = re.search(r"\bRR[:\s]*([0-9]{1,3})\b", t, re.I) or re.search(r"respiratory\s*rate\s*(is|:)?\s*([0-9]{1,3})\s*(/min|breaths?/min)", t, re.I)
    if m: clinical["rr.all"] = int(m.group(1))

    # SpO2
    m = re.search(r"(SpO2|SpO₂|sats?|oxygen|sat)[^\d]{0,6}([0-9]{2,3})\s*%?", t, re.I)
    if m: clinical["oxy.ra"] = int(m.group(2))

    # Labs
    for k in LAB_KEYS:
        m = re.search(fr"\b{k}\b[:\s]*(-?\d+(?:\.\d+)?)", t, re.I)
        if m:
            labs[k] = float(m.group(1))

    return clinical, labs, []

# ------------------------------
# Backend calls
# ------------------------------

def build_s1_payload(clinical_in: dict) -> dict:
    out = {}
    for k, v in (clinical_in or {}).items():
        if isinstance(v, str) and v.strip() == "":
            continue
        out[k] = v
    return out


def call_s1(clinical):
    payload = build_s1_payload(clinical)
    r = SESSION.post(API_S1, json={"features": payload},
                     timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_S1))
    r.raise_for_status()
    return r.json()

def call_s2(features, apply_calibration=True, allow_heavy_impute=False):
    payload = {"features": features, "apply_calibration": bool(apply_calibration)}
    if allow_heavy_impute:
        payload["allow_heavy_impute"] = True
    r = SESSION.post(API_S2, json=payload,
                     timeout=(CONNECT_TIMEOUT, READ_TIMEOUT_S2))
    r.raise_for_status()
    return r.json()

# ------------------------------
# Validation helpers (host-side; we never craft user text here)
# ------------------------------

def missing_for_s1(clinical: dict):
    return [k for k in S1_REQUIRED_MIN if k not in (clinical or {}) or (clinical or {}).get(k) in (None, "")]


def validated_set_name(features: dict) -> str | None:
    # work on a normalized copy so key casing/aliases don't matter
    feats = canonicalize_features({"labs": {k: v for k, v in (features or {}).items() if k not in ("age.months","sex")}, 
                                   "clinical": {"oxy.ra": features.get("oxy.ra")}})
    f = {**(feats.get("clinical") or {}), **(feats.get("labs") or {})}

    def provided(k):
        if k not in f: return False
        v = f[k]
        if v is None: return False
        if isinstance(v, str) and v.strip() == "": return False
        if v == 0 or v == 0.0: return False
        return True

    if provided("CRP") and provided("TNFR1") and provided("supar") and provided("oxy.ra"):
        return "A"
    if provided("CRP") and provided("CXCl10") and provided("IL6") and provided("oxy.ra"):
        return "B"
    lab_count = sum(1 for k in LAB_KEYS if provided(k))
    if lab_count >= 6:
        return "full_lab_panel"
    return None

# ------------------------------
# LLM Orchestration
# ------------------------------

def agent_call(user_text: str, sheet: dict, conv_id: str | None):
    context = {"sheet": sheet}
    input_items = [
        {"type": "message", "role": "system",
         "content": [{"type": "input_text", "text": AGENT_SYSTEM}]},
        {"type": "message", "role": "user",
         "content": [{"type": "input_text",
                      "text": f"CONTEXT\n{json.dumps(context, indent=2)}\n\nUSER\n{(user_text or '').strip()}"}]},
    ]

    resp = client.responses.create(
    model=_get_llm_model(),
    input=input_items,
    tools=TOOL_SPEC,        
    text={"verbosity": "medium"},
    reasoning={"effort": "medium"},
    parallel_tool_calls=False,
    max_tool_calls=1,         
    store=False,
    )

    say, cmd = "", None
    for item in (resp.output or []):
        if getattr(item, "type", "") == "message" and getattr(item, "role", "") == "assistant":
            for c in (getattr(item, "content", []) or []):
                if getattr(c, "type", "") == "output_text":
                    say = (getattr(c, "text", "") or "")
        if getattr(item, "type", "") in ("function_call", "tool_call") and getattr(item, "name", "") == "sepsis_command":
            try:
                cmd = json.loads(getattr(item, "arguments", "") or "{}")
            except Exception:
                cmd = None
    return (say.strip() or None), (cmd or None)


def agent_followup(sheet: dict, last_user: str = "", note: str = ""):
    user_text = (last_user or "").strip()
    if note:
        user_text += f"\n\n[system_note]: {note}"
    say2, _ = agent_call(user_text=user_text, sheet=sheet, conv_id=None)
    return say2 or ""

# ------------------------------
# Pipeline (host doesn’t craft dialogue)
# ------------------------------

def run_pipeline(state, user_text, use_llm=True):
    state = state or {"sheet": None}
    sheet = state.get("sheet") or new_sheet()

    have_key = len(os.getenv("OPENAI_API_KEY", "").strip()) >= 20
    if not use_llm or not have_key or PARSER_MODE != "llm_only":
        clin, labs, _ = extract_features(user_text or "")
        if clin or labs:
            sheet = merge_sheet(sheet, clin, labs)
            state["sheet"] = sheet
        # keep this terse; the LLM isn't in play here
        return state, "Noted. If this looks right, press **Run S1** or **Run S2**."

    say, cmd = agent_call(user_text=user_text, sheet=sheet, conv_id=None)

    updated = False
    if cmd and cmd.get("action") == "update_sheet":
        sheet = merge_features(sheet, cmd.get("features") or {})
        state["sheet"] = sheet
        updated = True

    # Only fall back if the model returned nothing.
    if say:
        return state, say

    # SMART host fallback: after updates, show availability or ask for missing
    if updated:
        guidance = build_guidance_after_update(state.get("sheet") or {})
        return state, guidance

    # Even if nothing changed, try to guide user if we can
    guidance = build_guidance_after_update(state.get("sheet") or {})
    if guidance:
        return state, guidance

    return state, "Okay."

def run_s1_click(history, st):
    sheet = st.get("sheet") or new_sheet()
    missing = missing_for_s1(sheet.get("features", {}).get("clinical", {}))
    if missing:
        msg = "Missing required fields for S1: " + ", ".join(missing) + "."
        history = history + [{"role": "assistant", "content": msg}]
        s1_upd, s2_upd = compute_btn_states(st)
        return history, st, json.dumps(sheet, indent=2), s1_upd, s2_upd

    try:
        s1 = call_s1(sheet["features"]["clinical"])
        sheet["s1"] = s1

        # meta-probs (robust to list/scalar)
        def _as_float(x):
            try: return float(_first(x))
            except: return None
        v1p = _as_float(((s1 or {}).get("v1") or {}).get("prob"))
        v2p = _as_float(((s1 or {}).get("v2") or {}).get("prob"))
        if v1p is not None:
            sheet["features"]["clinical"]["v1_pred_Severe"] = v1p
            sheet["features"]["clinical"]["v1_pred_Other"]  = 1.0 - v1p
        if v2p is not None:
            sheet["features"]["clinical"]["v2_pred_NOTSevere"] = v2p
            sheet["features"]["clinical"]["v2_pred_Other"]     = 1.0 - v2p

        st["sheet"] = sheet

        # Standardized message (no “next steps”)
        summary = format_s1_output(s1)
        history = history + [{"role": "assistant", "content": summary}]
        s1_upd, s2_upd = compute_btn_states(st)
        return history, st, json.dumps(sheet, indent=2), s1_upd, s2_upd

    except requests.Timeout:
        history = history + [{"role": "assistant",
                              "content": f"S1 timed out after {int(float(READ_TIMEOUT_S1))}s. "
                                         "The Info Sheet is unchanged. Try again or increase SEPSIS_API_READ_TIMEOUT_S1."}]
        s1_upd, s2_upd = compute_btn_states(st)
        return history, st, json.dumps(sheet, indent=2), s1_upd, s2_upd
    except Exception as e:
        history = history + [{"role": "assistant", "content": f"Error calling S1: {e}"}]
        s1_upd, s2_upd = compute_btn_states(st)
        return history, st, json.dumps(sheet, indent=2), s1_upd, s2_upd

def run_s2_click(history, st):
    sheet = st.get("sheet") or new_sheet()
    clin = sheet.get("features", {}).get("clinical", {})
    labs = sheet.get("features", {}).get("labs", {})
    merged = {**clin, **labs}

    vname = validated_set_name(merged)
    if vname is None and not st.get("awaiting_unvalidated_s2"):
        st["awaiting_unvalidated_s2"] = True
        warn = ("Warning: this biomarker combination is NOT VALIDATED. Results may be unreliable. "
                "Press **Run S2** again to proceed anyway, or add a validated set "
                "(A: CRP+TNFR1+suPAR+SpO₂ RA; B: CRP+CXCL10+IL6+SpO₂ RA).")
        history = history + [{"role": "assistant", "content": warn}]
        s1_upd, s2_upd = compute_btn_states(st)
        return history, st, json.dumps(sheet, indent=2), s1_upd, s2_upd

    try:
        s2 = call_s2(merged, apply_calibration=True)
        sheet["s2"] = s2
        st["sheet"] = sheet
        st["awaiting_unvalidated_s2"] = False

        summary = format_s2_output(s2)
        history = history + [{"role": "assistant", "content": summary}]
        s1_upd, s2_upd = compute_btn_states(st)
        return history, st, json.dumps(sheet, indent=2), s1_upd, s2_upd

    except requests.Timeout:
        history = history + [{"role": "assistant",
                              "content": f"S2 timed out after {int(float(READ_TIMEOUT_S2))}s. "
                                         "The Info Sheet is unchanged. Try again or increase SEPSIS_API_READ_TIMEOUT_S2."}]
        s1_upd, s2_upd = compute_btn_states(st)
        return history, st, json.dumps(sheet, indent=2), s1_upd, s2_upd
    except Exception as e:
        history = history + [{"role": "assistant", "content": f"Error calling S2: {e}"}]
        s1_upd, s2_upd = compute_btn_states(st)
        return history, st, json.dumps(sheet, indent=2), s1_upd, s2_upd

# ------------------------------
# Minimal UI (host never injects dialogue text)
# ------------------------------
SPACE_USER = os.getenv("SPACE_USER", "user")
SPACE_PASS = os.getenv("SPACE_PASS", "pass")


def check_login(u, p):
    ok = (u == SPACE_USER) and (p == SPACE_PASS)
    new_st = new_state() if ok else gr.update()
    return (
        gr.update(visible=not ok),
        gr.update(visible=ok),
        ("" if ok else "Invalid username or password."),
        new_st,
    )

with gr.Blocks(fill_height=True) as ui:
    # Login
    with gr.Group(visible=True) as login_view:
        gr.Markdown("#### 🔒 Sign in")
        u = gr.Textbox(label="Username", autofocus=True)
        p = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Enter")
        login_msg = gr.Markdown("")

    # App
    with gr.Group(visible=False) as app_view:
        gr.Markdown("### Spot Sepsis — Research Preview *(Not medical advice)*")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=420, type="messages")
                msg  = gr.Textbox(placeholder="Describe the case…", lines=3)
                use_llm_chk = gr.Checkbox(value=USE_LLM_DEFAULT, label="Use LLM to parse input (if available)", info="If unchecked, a lightweight built-in parser will extract basic fields only.")
                with gr.Row():
                    btn_send = gr.Button("Send Message")
                    btn_s1   = gr.Button("Run S1", interactive=False)   # start disabled
                    btn_s2   = gr.Button("Run S2", interactive=False)   # start disabled
                    btn_new  = gr.Button("New Chat")
            with gr.Column(scale=2):
                info = gr.Textbox(label="Current Info Sheet (JSON)", lines=22)
                paste = gr.Textbox(label="Paste an Info Sheet to restore/merge", lines=6)
                merge_btn = gr.Button("Merge")
                tips = gr.Markdown("")
                gr.Markdown(
                    "[Sepsis-Spotter-UI repo](https://github.com/ffr0517/Sepsis-Spotter-UI) · "
                    "[Sepsis-Spotter model repo](https://github.com/ffr0517/Sepsis-Spotter)\n\n"
                    "For more information please contact "
                    "[lcmrhodes98@gmail.com](mailto:lcmrhodes98@gmail.com)"
                )
                gr.Markdown(
                    "<div style='margin-top:8px; font-size:12px; color:#6b7280;'>"
                    "Timeouts may occur on first use if the backend is cold. If you encounter issues when first running S1 or S2, please try again after a few moments. "
                    "</div>"
                )

        state = gr.State(new_state())

        def new_chat_and_bootstrap():
            chat_reset, st, info_reset, paste_reset, tips_reset = reset_all()
            history = chat_reset + [{"role": "user", "content": ""}]
            st, reply = run_pipeline(st, "", use_llm=True)
            history = history + [{"role": "assistant", "content": reply}]
            info_json = json.dumps(st.get("sheet", {}), indent=2)
            return history, st, info_json, paste_reset, tips_reset

        def reset_all():
            return [], new_state(), "", "", "", gr.update(interactive=False), gr.update(interactive=False)

        def on_user_send(history, text):
            history = history + [{"role": "user", "content": text}]
            return history, ""
        
        def on_bot_reply(history, st, use_llm):
            st, reply = run_pipeline(st, history[-1]["content"], use_llm=bool(use_llm))
            history = history + [{"role": "assistant", "content": reply}]
            info_json = json.dumps(st.get("sheet", {}), indent=2)
            s1_upd, s2_upd = compute_btn_states(st)
            return history, st, info_json, "", s1_upd, s2_upd

        def on_merge(st, pasted):
            try:
                blob = json.loads(pasted)
            except Exception:
                s1_upd, s2_upd = compute_btn_states(st)
                return st, "Could not parse pasted JSON.", "", s1_upd, s2_upd
            if st.get("sheet"):
                st["sheet"] = merge_sheet(
                    st["sheet"],
                    blob.get("features", {}).get("clinical", {}),
                    blob.get("features", {}).get("labs", {})
                    )
            else:
                st["sheet"] = blob
            s1_upd, s2_upd = compute_btn_states(st)
            return st, "Merged.", json.dumps(st["sheet"], indent=2), s1_upd, s2_upd

        btn_s1.click(run_s1_click, [chat, state], [chat, state, info, btn_s1, btn_s2])
        btn_s2.click(run_s2_click, [chat, state], [chat, state, info, btn_s1, btn_s2])


        login_btn.click(check_login, [u, p], [login_view, app_view, login_msg, state])
        msg.submit(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, use_llm_chk], [chat, state, info, msg, btn_s1, btn_s2]
            )
        btn_send.click(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, use_llm_chk], [chat, state, info, msg, btn_s1, btn_s2]
            )
        merge_btn.click(on_merge, [state, paste], [state, tips, info, btn_s1, btn_s2])

        btn_new.click(reset_all, inputs=None, outputs=[chat, state, info, paste, tips, btn_s1, btn_s2])

IS_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SYSTEM") == "spaces")
if IS_SPACES:
    ui.launch(ssr_mode=False)
else:
    ui.launch(server_name="127.0.0.1", server_port=7860)

