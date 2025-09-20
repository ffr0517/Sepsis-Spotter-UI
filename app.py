import os, re, json, time, uuid
import requests
import gradio as gr


def new_state():
    # brand-new per-conversation state
    return {"sheet": None, "conv_id": None, "session": str(uuid.uuid4()), "awaiting_consent": False}


import logging, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("sepsis-agent")

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

USE_LLM_DEFAULT = True  # default for the UI checkbox

# ------------------------------
# Agent system prompt (LLM is the orchestrator)
# ------------------------------
AGENT_SYSTEM = """
You are Sepsis Spotter, a clinical intake and orchestration assistant (research preview; not a diagnosis).

## Mission & Style (STRICT)
- Help front-line clinicians use the Spot Sepsis models safely and efficiently.
- Be friendly, concise, and direct. Do not be verbose. No emojis.
- In your first natural message, display the disclaimer: “This is clinical decision support, not a diagnosis.”
- Never use raw variable names (such as rr.all); always use plain language (e.g., “breathing rate”); even when sending the 'pre-flight' confirmation, use plain language.
- Always end a message with a question asking for either more information, or confirmation to run the model.
- When emitting any tool call, you must also produce a short, user-visible message in the same turn. Never return a tool call with an empty user message.

## Operating Principles
- Never invent values. If unsure, ask a short clarifying question.
- One tool call per turn via `sepsis_command`, choosing exactly one of:
  • `ask` — request exactly one missing/high-impact field.
  • `update_sheet` — add values the user just supplied.
  • `call_api` — run S1 (or S2 if available).
- Do not restate all required fields unless something is missing.
- Do not paste the Info Sheet or JSON into the chat; the app UI shows state. Keep replies short.
- In `update_sheet`, only record values stated or confidently parsed from the user’s text. Do **not** insert placeholders in `update_sheet`. Placeholders are used **only** in `call_api` payloads.
- Tool discipline: One tool call per turn. If you just used update_sheet and essentials are present, the next turn should be a normal assistant message asking for consent (do not chain an immediate call_api in the same turn).

## Model Selection
- **S1**: always run first once clinical essentials are present (see S1 Payload Contract).
- **S2**: may run after S1 only if a **validated set** is present (plus S1→S2 meta-probs), otherwise warn once and require explicit confirmation to proceed unvalidated:
  • **Validated Set A**: CRP (mg/L), TNFR1 (pg/ml), suPAR (ng/ml), SpO₂ on room air (oxy.ra, %).
  • **Validated Set B**: CRP (mg/L), CXCl10 (pg/ml), IL-6 (pg/ml), SpO₂ on room air (oxy.ra, %).
  • **Full lab panel**: proceed if many labs are present; note features_sent: full_lab_panel in the info sheet.
  – If **no validated set** and user insists, warn: “Warning: this feature combination is NOT VALIDATED (see operational supplement). Results may be unreliable. Proceed only if you confirm.” If the user confirms, proceed and set validated_set=none (user-confirmed-unvalidated).
- If the user expresses **urgency**, you may run S1 immediately with placeholders (per S1 Payload Contract), then offer S2 if validated inputs exist.

## Pre-flight Confirmation (STRICT)
After any {"action":"update_sheet"}, if the minimal S1 validated set is present, do not call the API yet. Instead present a short pre-flight summary and ask for consent.

Minimal S1 set (all required):
• Age in months
• Sex (1 = male, 0 = female)
• Overnight hospitalisation within the last 6 months (1 = yes, 0 = no)
• Weight for age z-score
• Duration of illness (days)
• Not alert? (AVPU < A) (1 = yes, 0 = no)
• Heart rate (bpm)
• Respiratory rate (/min)
• Axillary temperature (°C)
• Capillary refill time >2 seconds? (1 = yes, 0 = no)

Pre-flight summary rules:
1) List what will be sent for S1 in plain language (never show raw keys).
2) Explicitly mark missing items as “omitted from the S1 call” (do NOT use 0/0.0 placeholders).
3) List any assumptions you made (e.g., “Assuming duration = 1 day based on ‘fever yesterday’.”).
4) End with one question asking for consent, e.g., “Shall I run S1 now?”

After the user explicitly consents (e.g., “yes”, “run S1”), your very next turn must be a single call_api tool call (stage “S1”). Do not send another update_sheet or a normal assistant message first.

Do NOT emit {"action":"call_api"} until the user consents.

## Intake & Validation
- Convert years→months for age.
- Map sex: **1 = male, 0 = female**.
- Range checks (gentle): HR 40–250, RR 10–120, SpO₂ 70–100 (%).
  - If any value is outside range, **flag every anomalous value at once** and ask for confirmation in one concise sentence (e.g., “HR 300, RR 8, and SpO₂ 105% look atypical — could you confirm these?”).
- When the user provides values in free text, you **must** emit `update_sheet` with **all parsed values at once** before asking another question.
- If approximating or assuming a value based on the user’s words (e.g., duration of illness from “fever yesterday”), explicitly state which values you have assumed.

## Interaction Flow
- **First user turn** and sheet empty → send this exact first message (no extra text before or after):

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

- If the input **omits essentials**, emit a single `ask` that gently nudges for the missing pieces.
- When essentials are present, emit `call_api` with the **full S1 payload** (fill unknowns with placeholders as specified below).

## S1 Payload Contract (Strict)
When emitting {"action":"call_api","stage":"S1"}, include features.clinical with ONLY the fields the user actually provided. Omit unknowns entirely. Never fabricate values.
- Sex encoding: 1 = male, 0 = female (use only if the user provided sex).
- For any field not provided by the user: do not send a placeholder.
- Use numbers for continuous values and 0/1 for binary flags **only when stated by the user**.

### Field dictionary (key → meaning → type → placeholder) 
*NOTE THAT KEY VALUES MUST NEVER BE EXPOSED TO THE USER*
- `age.months` → Age in months → number → 0.0
- `sex` → Sex (1=male, 0=female) → integer {0,1} → 0
- `bgcombyn` → Comorbidity present → integer {0,1} → 0
- `adm.recent` → Overnight hospitalisation within the last 6 months (1 = yes, 0 = no) → integer {0,1} → 0
- `wfaz` → Weight-for-age Z-score → number → 0.0
- `waste` → Wasting (WFL Z < −2) → integer {0,1} → 0
- `stunt` → Stunting (LAZ < −2) → integer {0,1} → 0
- `cidysymp` → Duration of illness (days) → integer ≥0 → 0
- `prior.care` → Prior care-seeking → integer {0,1} → 0
- `travel.time.bin` → Travel time ≤1h (1=yes, 0=>1h) → integer {0,1} → 0
- `diarrhoeal` → Diarrhoeal syndrome → integer {0,1} → 0
- `pneumo` → WHO pneumonia → integer {0,1} → 0
- `sev.pneumo` → WHO severe pneumonia → integer {0,1} → 0
- `ensapro` → Prostration/encephalopathy → integer {0,1} → 0
- `vomit.all` → Intractable vomiting → integer {0,1} → 0
- `seiz` → Convulsions → integer {0,1} → 0
- `pfacleth` → Lethargy → integer {0,1} → 0
- `not.alert` → Not alert (AVPU < A) → integer {0,1} → 0
- `danger.sign` → Any IMCI danger sign → integer {0,1} → 0
- `hr.all` → Heart rate (bpm) → number → 0.0
- `rr.all` → Respiratory rate (breaths/min) → number → 0.0
- `oxy.ra` → SpO₂ on room air (%) → number → 0.0
- `envhtemp` → Axillary temperature (°C) → number → 0.0
- `crt.long` → Capillary refill >2 s → integer {0,1} → 0
- `parenteral_screen` → Parenteral treatment before enrolment → integer {0,1} → 0
- `SIRS_num` → SIRS score (0–4) → integer 0–4 → 0

## Canonical S1 `call_api` Template
{
  "action": "call_api",
  "stage": "S1",
  "message": "Running S1 now.",
  "features": {
    "clinical": {
      "age.months": 0.0,
      "sex": 0,
      "bgcombyn": 0,
      "adm.recent": 0,
      "wfaz": 0.0,
      "waste": 0,
      "stunt": 0,
      "cidysymp": 0,
      "prior.care": 0,
      "travel.time.bin": 0,
      "diarrhoeal": 0,
      "pneumo": 0,
      "sev.pneumo": 0,
      "ensapro": 0,
      "vomit.all": 0,
      "seiz": 0,
      "pfacleth": 0,
      "not.alert": 0,
      "danger.sign": 0,
      "hr.all": 0.0,
      "rr.all": 0.0,
      "oxy.ra": 0.0,
      "envhtemp": 0.0,
      "crt.long": 0,
      "parenteral_screen": 0,
      "SIRS_num": 0
    }
  }
}

## Output Formatting for Proving Results to the User
After receiving an API response with a decision, always present the prediction using this standard format:

- If **Severe**:  
  “S1 decision: SEVERE. According to historical data and model specifics, the given patient’s symptoms suggest a severe outcome within 48 hours. That is, death/receipt of organ support/discharged home to die within 48 hours.”

- If **NOTSevere**:  
  “S1 decision: NOT SEVERE. According to historical data and model specifics, the given patient’s symptoms suggest a non-severe disease. That is, no admittance to any health facility, and symptoms resolved within 28 days.”

- If **Other**:  
  “S1 decision: OTHER. According to historical data and model specifics, laboratory tests/biomarkers are required to make a more informed outcome prediction. Please note that the model incorporating laboratory results and biomarkers is NOT currently available.”

Always follow with the disclaimer:
“This is clinical decision support, not a diagnosis. You must use your own clinical judgment, training, and knowledge to make referral or treatment decisions. No liability is accepted.”

## Error Handling
- If an API call fails **due to timeout**, respond:
  “The model API did not respond in time. Please try again in about 60 seconds.”
- Do **not** invent results or repeat a stale prediction.
- Keep the sheet state unchanged until a successful API response is received.

## Worked Example
{
  "action": "call_api",
  "stage": "S1",
  "message": "Running S1 now.",
  "features": {
    "clinical": {
      "age.months": 24.0,
      "sex": 1,
      "bgcombyn": 0,
      "adm.recent": 0,
      "wfaz": -1.2,
      "waste": 0,
      "stunt": 0,
      "cidysymp": 2,
      "prior.care": 0,
      "travel.time.bin": 0,
      "diarrhoeal": 0,
      "pneumo": 0,
      "sev.pneumo": 0,
      "ensapro": 0,
      "vomit.all": 0,
      "seiz": 0,
      "pfacleth": 0,
      "not.alert": 0,
      "danger.sign": 0,
      "hr.all": 128.0,
      "rr.all": 32.0,
      "oxy.ra": 96.0,
      "envhtemp": 28.0,
      "crt.long": 0,
      "parenteral_screen": 0,
      "SIRS_num": 1
    }
  }
}

## Pre-flight summary template
“Here’s what I will send to the model:
• Age: [VALUE] months; Sex: [GENDER]; Heart rate: [VALUE] bpm; Breathing rate: [VALUE] /min; Oxygen on room air: [VALUE]%; Temperature: [VALUE] °C; Alertness: [VALUE]; Capillary refill: [VALUE]; IMCI danger signs: [VALUE]; Severe pneumonia: [VALUE]; Vomiting: [VALUE]; Seizure: [VALUE]; Diarrhoea: [VALUE]; Comorbidity: [VALUE].
Unknown (omitted from the S1 call):[ANY MISSING VALUES]
Assumptions: [DESCRIPTION OF ANY ASSUMED VALUES].
Shall I run S1 now?”

Remember: You are an orchestrator, not a decision-maker. Collect inputs, validate, run the model, and return clear, auditable results.
"""

TOOL_SPEC = [{
    "type": "function",
    "name": "sepsis_command",                     # <- top-level name (required)
    "description": "Single structured command: ask user, update sheet, or call API.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["ask", "update_sheet", "call_api"]},
            "message": {"type": "string", "description": "Short user-visible text/question."},
            "features": {
                "type": "object",
                "properties": {
                    "clinical": {"type": "object", "additionalProperties": {"type": ["number","integer","string","boolean"]}},
                    "labs": {"type": "object", "additionalProperties": {"type": ["number","integer","string","boolean"]}}
                }
            },
            "stage": {"type": "string", "enum": ["auto","S1","S2"]}
        },
        "required": ["action"],
        "additionalProperties": False
    }
}]

# --------------------------------
# Config via env (set in Spaces → Settings → Variables)
# --------------------------------
S1_FIELDS = [
    "age.months","sex","bgcombyn","adm.recent","wfaz","waste","stunt",
    "cidysymp","prior.care","travel.time.bin","diarrhoeal","pneumo","sev.pneumo",
    "ensapro","vomit.all","seiz","pfacleth","not.alert","danger.sign",
    "hr.all","rr.all","oxy.ra","envhtemp","crt.long","parenteral_screen","SIRS_num"
]

# Gate/ask list = validated minimal set for S1 (exact spec)
S1_REQUIRED_MIN = [
    "age.months",   # Age in months
    "sex",          # 1=male, 0=female
    "adm.recent",   # Overnight hospitalisation last 6 months (1/0)
    "wfaz",         # Weight-for-age z-score
    "cidysymp",     # Duration of illness (days)
    "not.alert",    # AVPU < A (1/0)
    "hr.all",       # Heart rate (bpm)
    "rr.all",       # Respiratory rate (/min)
    "envhtemp",     # Axillary temperature (°C)
    "crt.long"      # Capillary refill >2 s (1/0)
]

# Human-readable labels for prompts
S1_REQUIRED_LABELS = {
    "age.months": "age (months)",
    "sex": "sex (1 = male, 0 = female)",
    "adm.recent": "overnight hospitalisation within the last 6 months (1 = yes, 0 = no)",
    "wfaz": "weight-for-age z-score",
    "cidysymp": "duration of illness (days)",
    "not.alert": "not alert? (AVPU < A) (1 = yes, 0 = no)",
    "hr.all": "heart rate (bpm)",
    "rr.all": "breathing rate (/min)",
    "envhtemp": "axillary temperature (°C)",
    "crt.long": "capillary refill time greater than 2 seconds? (1 = yes, 0 = no)",
}

def humanize_field(key: str) -> str:
    return S1_REQUIRED_LABELS.get(key, key.replace(".", " "))

def build_s1_payload(clinical_in: dict) -> dict:
    """
    For S1, send ONLY the keys the user actually provided.
    No placeholders. The API will align schema and impute as needed.
    """
    out = {}
    for k, v in (clinical_in or {}).items():
        # light cast: keep ints for flags, floats for continuous if they look numeric
        if isinstance(v, str) and v.strip() == "":
            continue
        out[k] = v
    return out

API_S1 = os.getenv("SEPSIS_API_URL_S1", "https://sepsis-spotter-beta.onrender.com/s1_infer")
API_S2 = os.getenv("SEPSIS_API_URL_S2", "https://sepsis-spotter-beta.onrender.com/s2_infer")

# S2 lab keys (names/units per your spec)
LAB_KEYS = [
    "CRP", "TNFR1", "supar", "CXCl10", "IL6", "IL10", "IL1ra", "IL8", "PROC",
    "ANG1", "ANG2", "CHI3L", "STREM1", "VEGFR1", "lblac", "lbglu", "enescbchb1"
]

def validate_complete(clinical: dict):
    """
    Gate only on the validated minimal S1 set; add light, non-blocking range checks.
    """
    missing = [k for k in S1_REQUIRED_MIN if k not in clinical]
    warnings = []

    # Range checks (gentle; do not block)
    try:
        if "hr.all" in clinical and not (40 <= float(clinical["hr.all"]) <= 250):
            warnings.append("Heart rate seems out of range.")
    except Exception:
        pass
    try:
        if "rr.all" in clinical and not (10 <= float(clinical["rr.all"]) <= 120):
            warnings.append("Breathing rate seems out of range.")
    except Exception:
        pass
    try:
        if "envhtemp" in clinical and not (30 <= float(clinical["envhtemp"]) <= 43):
            warnings.append("Temperature seems out of range.")
    except Exception:
        pass
    try:
        if "age.months" in clinical and not (0 <= float(clinical["age.months"]) <= 180):
            warnings.append("Age (months) seems out of range.")
    except Exception:
        pass

    return missing, warnings

# --------------------------------
# Simple rule-based extractor (robust + free) for legacy path
# --------------------------------
def extract_features(text: str):
    clinical, labs, notes = {}, {}, []
    t = text or ""

    # age: years or months
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)\b", t, re.I)
    if m: clinical["age.months"] = float(m.group(1)) * 12
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?|mo)\b", t, re.I)
    if m: clinical["age.months"] = float(m.group(1))

    # sex
    if re.search(r"\bmale\b|\bboy\b", t, re.I): clinical["sex"] = 1
    if re.search(r"\bfemale\b|\bgirl\b", t, re.I): clinical["sex"] = 0

    # vitals
    m = re.search(r"\bHR[:\s]*([0-9]{2,3})\b", t, re.I)
    if m: clinical["hr.all"] = int(m.group(1))
    m = re.search(r"\bRR[:\s]*([0-9]{1,3})\b", t, re.I)
    if m: clinical["rr.all"] = int(m.group(1))

    # --- SpO2 / oxygen saturation ---
    # Case A: term first → number after (e.g., "SpO2 96", "oxygen saturation 96%")
    m = re.search(r"(?:SpO2|SpO₂|sats?|oxygen(?:\s+saturation)?)[:\s]*([0-9]{2,3})(?:\s*%)?\b", t, re.I)
    if m:
        clinical["oxy.ra"] = int(m.group(1))
    else:
        # Case B: number first → term after (e.g., "96% oxygen saturation", "96% oxygen")
        m = re.search(r"\b([0-9]{2,3})\s*%(?:\s*(?:on\s+room\s+air|RA))?\s*(?:oxygen(?:\s+saturation)?|SpO2|SpO₂|sats?)\b", t, re.I)
        if m:
            clinical["oxy.ra"] = int(m.group(1))

    # common flags
    if re.search(r"\bdanger sign(s)?\b", t, re.I): clinical["danger.sign"] = 1
    if re.search(r"\bnot alert\b|\bletharg(y|ic)\b|\bdrows(y|iness)\b", t, re.I): clinical["not.alert"] = 1
    if re.search(r"\bURTI\b|\bupper resp", t, re.I): clinical["urti"] = 1
    if re.search(r"\bLRTI\b|\blower resp|\bpneumonia\b", t, re.I): clinical["lrti"] = 1
    if re.search(r"\bdiarrh", t, re.I): clinical["diarrhoeal"] = 1
    m = re.search(r"\bcrt[:\s]*([0-9](?:\.[0-9])?)\b", t, re.I)
    if m: clinical["crt.long"] = 1 if float(m.group(1)) >= 3 else 0

    # --- Labs with aliases ---
    # Map many phrasings to canonical keys used by your API
    LAB_ALIASES = [
        # canonical,            regex that matches in text
        ("CRP",                 r"\b(?:CRP|C[-\s]*reactive\s*protein)\b"),
        ("CXCl10",              r"\b(?:CXCL?10)\b"),               # tolerate CXCL10 or CXCl10
        ("IL6",                 r"\b(?:IL[-\s]*6|Interleukin[-\s]*6)\b"),
        ("IL10",                r"\b(?:IL[-\s]*10|Interleukin[-\s]*10)\b"),
        ("IL1ra",               r"\b(?:IL[-\s]*1RA|IL1RA|IL[-\s]*1\s*receptor\s*antagonist)\b"),
        ("IL8",                 r"\b(?:IL[-\s]*8|Interleukin[-\s]*8)\b"),
        ("PROC",                r"\b(?:PCT|Procalcitonin)\b"),
        ("TNFR1",               r"\b(?:TNF\s*R?1|TNFR1|s?TNF[-\s]*receptor[-\s]*1)\b"),
        ("supar",               r"\b(?:su?PAR)\b"),
        ("ANG1",                r"\bANG(?:I|1)\b"),
        ("ANG2",                r"\bANG(?:II|2)\b"),
        ("CHI3L",               r"\bCHI3L1?\b"),
        ("STREM1",              r"\bs?TREM[-\s]*1\b"),
        ("VEGFR1",              r"\bVEGFR[-\s]*1\b"),
        ("lblac",               r"\b(?:Lactate)\b"),
        ("lbglu",               r"\b(?:Glucose)\b"),
        ("enescbchb1",          r"\b(?:Haemoglobin|Hemoglobin|Hgb)\b"),
    ]

    # helper to pull the first numeric value after a matched alias
    def _grab_value_after(pattern: str):
        # match "... <alias> : 12.3" or "... <alias> = 12.3" or "... <alias> 12.3"
        m = re.search(pattern + r"[^0-9\-]*(-?[0-9]+(?:\.[0-9]+)?)", t, re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
        return None

    for canon, alias_rx in LAB_ALIASES:
        val = _grab_value_after(alias_rx)
        if val is not None:
            labs[canon] = val

    return clinical, labs, notes

def validate_complete(clinical: dict):
    """
    Return (missing, warnings) where:
      - missing: keys from the validated minimal S1 set that are absent
      - warnings: gentle range checks (non-blocking)
    """
    # Gate ONLY on the validated minimal S1 set
    missing = [k for k in S1_REQUIRED_MIN if k not in clinical or clinical[k] in (None, "")]
    warnings = []

    # Gentle range checks (do not block)
    try:
        if "hr.all" in clinical and clinical["hr.all"] not in (None, ""):
            if not (40 <= float(clinical["hr.all"]) <= 250):
                warnings.append("Heart rate seems out of range.")
    except Exception:
        pass

    try:
        if "rr.all" in clinical and clinical["rr.all"] not in (None, ""):
            if not (10 <= float(clinical["rr.all"]) <= 120):
                warnings.append("Breathing rate seems out of range.")
    except Exception:
        pass

    try:
        if "envhtemp" in clinical and clinical["envhtemp"] not in (None, ""):
            if not (30 <= float(clinical["envhtemp"]) <= 43):
                warnings.append("Temperature seems out of range.")
    except Exception:
        pass

    try:
        if "age.months" in clinical and clinical["age.months"] not in (None, ""):
            if not (0 <= float(clinical["age.months"]) <= 180):
                warnings.append("Age (months) seems out of range.")
    except Exception:
        pass

    # Optional: warn on SpO₂ if present (not required for S1 gating)
    try:
        if "oxy.ra" in clinical and clinical["oxy.ra"] not in (None, ""):
            if not (70 <= float(clinical["oxy.ra"]) <= 100):
                warnings.append("SpO₂ seems out of range.")
    except Exception:
        pass

    return missing, warnings

# --------------------------------
# Info sheet helpers
# --------------------------------
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

def _normalize_decision(val):
    # Accept "Severe", "NOTSevere", "Other", ["Severe"], etc.
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    val = (val or "").strip()
    return val

S1_DISCLAIMER = (
    "This is clinical decision support, not a diagnosis. You must use your own clinical judgment, "
    "training, and knowledge to make referral or treatment decisions. No liability is accepted."
)

def format_s1_decision(decision):
    d = _normalize_decision(decision)
    # normalize case but keep exact matching keys
    key = d.replace(" ", "").upper()
    if key == "SEVERE":
        body = ("S1 decision: SEVERE. According to historical data and model specifics, the given "
                "patient’s symptoms suggest a severe outcome within 48 hours. That is, death/receipt of "
                "organ support/discharged home to die within 48 hours.")
    elif key in ("NOTSEVERE", "NOTSEVERE."):  # tolerate punctuation
        body = ("S1 decision: NOT SEVERE. According to historical data and model specifics, the given "
                "patient’s symptoms suggest a non-severe disease. That is, no admittance to any health "
                "facility, and symptoms resolved within 28 days.")
    else:
        # default to OTHER if unknown
        body = ("S1 decision: OTHER. According to historical data and model specifics, laboratory "
                "tests/biomarkers are required to make a more informed outcome prediction. Please note "
                "that the model incorporating laboratory results and biomarkers is NOT currently available.")
    return f"{body}\n\n{S1_DISCLAIMER}"

def _compute_meta_from_s1(s1_json: dict) -> dict:
    """
    Compute S1→S2 meta-probabilities from /s1_infer response:
      v1_pred_Severe = v1.prob; v1_pred_Other = 1 - v1.prob;
      v2_pred_NOTSevere = v2.prob; v2_pred_Other = 1 - v2.prob.
    """
    try:
        v1p = float(s1_json.get("v1", {}).get("prob"))
    except Exception:
        v1p = None
    try:
        v2p = float(s1_json.get("v2", {}).get("prob"))
    except Exception:
        v2p = None

    out = {}
    if v1p is not None:
        out["v1_pred_Severe"] = v1p
        out["v1_pred_Other"] = 1.0 - v1p
    if v2p is not None:
        out["v2_pred_NOTSevere"] = v2p
        out["v2_pred_Other"] = 1.0 - v2p
    return out


def _validated_set_name(features: dict) -> str | None:
    """
    Return 'A', 'B', 'full_lab_panel', or None based on available features.
    Requires non-placeholder oxy.ra for A/B (not 0/0.0/empty/None).
    """
    f = features or {}

    def provided(k):
        if k not in f:
            return False
        v = f[k]
        if v is None:
            return False
        if isinstance(v, str) and v.strip() == "":
            return False
        # Treat explicit placeholders as not provided
        if v == 0 or v == 0.0:
            return False
        return True

    # Validated Set A
    if provided("CRP") and provided("TNFR1") and provided("supar") and provided("oxy.ra"):
        return "A"

    # Validated Set B
    if provided("CRP") and provided("CXCl10") and provided("IL6") and provided("oxy.ra"):
        return "B"

    # Heuristic: "full lab panel" if many labs (≥6) present
    LAB_KEYS = [
        "CRP","TNFR1","supar","CXCl10","IL6","IL10","IL1ra","IL8","PROC",
        "ANG1","ANG2","CHI3L","STREM1","VEGFR1","lblac","lbglu","enescbchb1"
    ]
    lab_count = sum(1 for k in LAB_KEYS if provided(k))
    if lab_count >= 6:
        return "full_lab_panel"

    return None


def _extract_s2_call(s2_json):
    """
    /s2_infer returns a list with one row containing 'call' (4-class decision).
    This extracts it safely.
    """
    try:
        if isinstance(s2_json, list) and s2_json:
            return str(s2_json[0].get("call") or "")
    except Exception:
        pass
    try:
        return str(s2_json.get("call") or "")
    except Exception:
        return ""

# --------------------------------
# Model calls
# --------------------------------
def call_s1(clinical):
    try:
        payload = build_s1_payload(clinical)
        r = requests.post(API_S1, json={"features": payload}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"S1 endpoint unavailable or returned an error: {e}") from e


def call_s2(features, apply_calibration=True, allow_heavy_impute=False):
    try:
        payload = {
            "features": features,
            "apply_calibration": bool(apply_calibration),
        }
        if allow_heavy_impute:
            payload["allow_heavy_impute"] = True
        r = requests.post(API_S2, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        # Try to surface useful reason text from server (e.g., 422 with 'reason')
        try:
            detail = r.json()
        except Exception:
            detail = {}
        reason = detail.get("reason") or detail.get("error") or ""
        raise RuntimeError(f"S2 error: {reason or e}") from e

# --------------------------------
# Agent helpers (LLM orchestrator)
# --------------------------------
def _get_llm_model():
    m = os.getenv("LLM_MODEL_ID", "").strip()
    if not m or "/" in m:
        return "gpt-4o-mini"
    return m



# --------------------------------
# Orchestration (Agent-first when toggle ON)
# --------------------------------
DEBUG_AGENT = bool(int(os.getenv("DEBUG_AGENT", "0")))

def llm_available() -> bool:
    """Return True iff we have a plausible OpenAI API key in env."""
    k = os.getenv("OPENAI_API_KEY", "").strip()
    return len(k) >= 20  # cheap sanity check; avoids empty/placeholder keys


def safe_agent_step(user_text: str, sheet: dict, conv_id: str | None):
    """
    Wrap agent_step in a try/except so we never break the UI.
    Returns: (say, cmds, conv_id, error_text_or_None)
    """
    try:
        say, cmds, new_conv = agent_step(user_text, sheet, conv_id)
        return say, cmds, new_conv, None
    except Exception as e:
        log.exception("[AGENT] agent_step failed")
        return None, [], conv_id, f"Agent mode error: {e}"

def run_pipeline(state, user_text, stage="auto", use_llm=True):
    """
    Entry used by the Gradio callbacks.
    Returns (state, reply_text)
    """
    state = state or {"sheet": None, "conv_id": None}
    sheet = state.get("sheet") or new_sheet()

    # Always parse/merge from user text first
    clin_new, labs_new, _ = extract_features(user_text or "")
    if clin_new or labs_new:
        sheet = merge_sheet(sheet, clin_new, labs_new)
        state["sheet"] = sheet

    # If user disabled LLM or API key missing, go legacy immediately
    if not use_llm or not llm_available():
        if not use_llm:
            log.info("[AGENT] Skipping LLM (checkbox off); using legacy pipeline.")
        else:
            log.warning("[AGENT] OPENAI_API_KEY missing/invalid; using legacy pipeline.")
        return run_pipeline_legacy(state, user_text, stage=stage)

    # Try the agent (LLM) path, and cleanly fall back if anything goes wrong
    say, cmds, new_conv, err = safe_agent_step(user_text, sheet, state.get("conv_id"))

    if err:
        # Fall back to legacy, but tell the user what happened (briefly)
        state, reply = run_pipeline_legacy(state, user_text, stage=stage)
        reply = "Agent mode is temporarily unavailable — continuing in direct mode.\n\n" + reply
        return state, reply

    # If the agent returned tool calls, apply them in order
    reply = say or ""
    if cmds:
        for cmd in cmds:
            state, reply = handle_tool_cmd(state, cmd, user_text, stage_hint=stage)
    elif not reply:
        reply = "Ok."

    return state, reply


def handle_tool_cmd(state, cmd, user_text, stage_hint="auto"):
    """
    Applies a sepsis_command dict to state, returns (state, reply_text).
    """
    sheet = state.get("sheet") or new_sheet()

    action = (cmd or {}).get("action")
    message = (cmd or {}).get("message") or ""

    # --- Consent & essentials gate ---
    consent_patterns = r"\b(yes|yep|yeah|y|ok|okay|proceed|go ahead|run( the)? model|run s1|call s1|do it|please run)\b"
    user_ok = bool(re.search(consent_patterns, (user_text or "").lower()))

    try:
        missing_req, _warn = validate_complete((sheet.get("features") or {}).get("clinical", {}))
        essentials_present = (len(missing_req) == 0)
    except Exception:
        essentials_present = False

    def _run_s1_now():
        # helper to actually run S1 and clear consent gate
        miss, _warn = validate_complete(sheet["features"]["clinical"])
        if miss:
            return state, "Missing required fields for S1: " + ", ".join(miss) + "."
        try:
            s1 = call_s1(sheet["features"]["clinical"])
            sheet["s1"] = s1
            # Store meta-probs for S2
            sheet.setdefault("features", {}).setdefault("clinical", {}).update(_compute_meta_from_s1(s1))
            state["sheet"] = sheet
            state["awaiting_consent"] = False
            reply = (message or "Running S1 now.") + "\n\n" + format_s1_decision(s1.get("s1_decision"))
            return state, reply
        except Exception as e:
            return state, f"Error calling S1 API: {e}"

    # Short-circuit: if user explicitly consented and essentials exist, run S1 now
    if user_ok and essentials_present and action != "call_api":
        return _run_s1_now()

    if state.get("awaiting_consent") and essentials_present and action in ("ask", "update_sheet") and not user_ok:
        return state, (message or "I have what I need.") + "\n\nShall I run S1 now?"

    # ---- Actions ----
    if action == "ask":
        state["sheet"] = sheet
        return state, (message or "Could you clarify one detail?")

    if action == "update_sheet":
        feats = (cmd or {}).get("features") or {}
        sheet = merge_features(sheet, feats)
        state["sheet"] = sheet

        # Determine next prompt and consent status
        missing, warnings = validate_complete(sheet["features"]["clinical"])
        state["awaiting_consent"] = (len(missing) == 0)

        if stage_hint in ("S1", "auto"):
            if missing:
                next_required = missing[0]
                nxt = f"{message}\n\nCould you provide **{next_required}**?"
            else:
                nxt = message or "Thanks — I have the essentials."
        else:
            nxt = message or "Got it."

        if warnings:
            nxt += "\n\n⚠️ " + " ".join(warnings)

        if state["awaiting_consent"]:
            nxt = (message or "I have what I need.") + "\n\nShall I run S1 now?"

        return state, nxt

    if action == "call_api":
        # Require explicit consent if we weren't already awaiting it
        user_ok = bool(re.search(r"\b(yes|run|proceed|go ahead|call s1|run s1)\b", (user_text or "").lower()))
        if not state.get("awaiting_consent") and not user_ok:
            return state, "I’m ready to run S1. Please confirm: shall I run it now?"

        stage = (cmd or {}).get("stage") or "auto"
        # Merge any features the tool call included
        feats = (cmd or {}).get("features") or {}
        sheet = merge_features(sheet, feats)
        state["sheet"] = sheet
        state["awaiting_consent"] = False

        # Choose stage
        if stage == "auto":
            stage = "S2" if sheet["features"]["labs"] else "S1"

        if stage == "S1":
            missing, _warnings = validate_complete(sheet["features"]["clinical"])
            if missing:
                return state, "Missing required fields for S1: " + ", ".join(missing) + "."
            return _run_s1_now()

        elif stage == "S2":
            # Ensure S1 has been run to obtain meta-probs first
            if "s1" not in sheet:
                miss, _ = validate_complete(sheet["features"]["clinical"])
                if miss:
                    return state, ("Before S2, we need to run S1 to obtain required meta-probabilities. "
                                   "Missing: " + ", ".join(miss) + ".")
                try:
                    s1 = call_s1(sheet["features"]["clinical"])
                    sheet["s1"] = s1
                    sheet["features"]["clinical"].update(_compute_meta_from_s1(s1))
                except Exception as e:
                    return state, f"Error calling S1 (required before S2): {e}"

            # Merge clinical + labs
            merged = {**sheet["features"]["clinical"], **sheet["features"]["labs"]}

            # Validated set policy
            vname = _validated_set_name(merged)
            if vname is None and not re.search(r"\b(yes|proceed|confirm|go ahead)\b", (user_text or "").lower()):
                state["awaiting_consent"] = True
                return state, ("Warning: this feature combination is NOT VALIDATED (see operational supplement). "
                               "Results may be unreliable. Proceed only if you confirm.")

            try:
                s2 = call_s2(merged, apply_calibration=True)
                sheet["s2"] = s2
                state["sheet"] = sheet
                s2_call = _extract_s2_call(s2) or "Unavailable"
                reply = (message or "Running S2 now.") + f"\n\n**S2 decision:** {s2_call}"
                return state, reply
            except Exception as e:
                return state, f"Error calling S2 API: {e}"

        else:
            return state, f"Unknown stage: {stage}"

    # Fallback if the tool payload is malformed
    state["sheet"] = sheet
    return state, "I didn’t receive a valid action from the tool call."


def run_pipeline_legacy(state, user_text, stage="auto"):
    clin_new, labs_new, _ = extract_features(user_text or "")

    sheet = state.get("sheet") or new_sheet()
    sheet = merge_sheet(sheet, clin_new, labs_new)

    if stage == "auto":
        stage = "S2" if sheet["features"]["labs"] else "S1"

    missing, _ = validate_complete(sheet["features"]["clinical"])
    if stage == "S1" and missing:
        msg = "Missing required fields for S1: " + ", ".join(missing) + ". Please provide them."
        state["sheet"] = sheet
        return state, msg

    try:
        if stage == "S1":
            s1 = call_s1(sheet["features"]["clinical"])
            sheet["s1"] = s1
            sheet["features"]["clinical"].update(_compute_meta_from_s1(s1))
            state["sheet"] = sheet
            state["awaiting_consent"] = False
            reply = "Running S1 now.\n\n" + format_s1_decision(s1.get("s1_decision"))
            return state, reply
        elif stage == "S2":
            features = {**sheet["features"]["clinical"], **sheet["features"]["labs"]}
            s2 = call_s2(features, apply_calibration=True)
            sheet["s2"] = s2
            state["sheet"] = sheet
            state["awaiting_consent"] = False
            s2_call = _extract_s2_call(s2) or "Unavailable"
            reply = "Running S2 now.\n\n**S2 decision:** " + s2_call
            return state, reply
        else:
            return state, "Unknown stage."
    except Exception as e:
        return state, f"Error calling API: {e}"




def plausible_from_text(text, k, v):
    # very light check: does a number appear that matches v?
    if isinstance(v, (int, float)):
        return re.search(rf"\b{int(v)}\b", text or "") is not None
    if k == "sex":
        return bool(re.search(r"\b(male|boy|female|girl)\b", text or "", re.I))
    return True

def merge_features(sheet, feats):
    return merge_sheet(
        sheet,
        (feats or {}).get("clinical", {}) or {},
        (feats or {}).get("labs", {}) or {},
    )


def agent_step(user_text: str, sheet: dict | None, conv_id: str | None):
    """
    Ask the model to chat naturally AND optionally emit one or more structured tool calls.
    Returns: (say_text, cmds_list_in_order, new_conv_id_or_original)
    """
    sheet = sheet or new_sheet()
    context = {"sheet": sheet}

    # Messages: content entries must be "input_text" for inputs
    input_items = [
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": AGENT_SYSTEM}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": f"CONTEXT:\n{json.dumps(context, indent=2)}\n\nUSER:\n{(user_text or '').strip()}",
            }],
        },
    ]

    resp = client.responses.create(
        model=_get_llm_model(),
        input=input_items,
        tools=[{
            "type": "function",
            "name": "sepsis_command",
            "description": "Single structured command: ask user, update sheet, or call API.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["ask", "update_sheet", "call_api"]},
                    "message": {"type": "string"},
                    "features": {
                        "type": "object",
                        "properties": {
                            "clinical": {"type": "object", "additionalProperties": True},
                            "labs": {"type": "object", "additionalProperties": True},
                        },
                    },
                    "stage": {"type": "string", "enum": ["auto", "S1", "S2"]},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }],
        text={"verbosity": "low"},          # "low" | "medium" | "high"
        reasoning={"effort": "low"},        # "low" | "medium" | "high"
        parallel_tool_calls=False,   # <— add
        max_tool_calls=1,            # <— add
        store=False
)


    say = ""
    cmds = []

    for item in (resp.output or []):
        # Assistant text (if any)
        if getattr(item, "type", "") == "message" and getattr(item, "role", "") == "assistant":
            for c in (getattr(item, "content", []) or []):
                if getattr(c, "type", "") == "output_text":
                    say = (getattr(c, "text", "") or "")

        # Tool calls (may be multiple)
        if getattr(item, "type", "") in ("function_call", "tool_call") and getattr(item, "name", "") == "sepsis_command":
            try:
                cmds.append(json.loads(getattr(item, "arguments", "") or "{}"))
            except Exception:
                if DEBUG_AGENT:
                    log.info("[RESPONSES TOOL ARGS PARSE ERROR] %s", getattr(item, "arguments", ""))

    if DEBUG_AGENT:
        try:
            log.info("[RESPONSES RAW]\n%s", resp.model_dump_json(indent=2))
        except Exception:
            log.info("[RESPONSES RAW] %s", resp)
        log.info("[RESPONSES SAY] %s", say.strip())
        log.info("[RESPONSES CMDS] %s", json.dumps(cmds, indent=2))
        log.info("[SHEET] %s", json.dumps(sheet, indent=2))

    return (say.strip() or None), cmds, conv_id


# --------------------------------
# In-app Login + Gradio UI (uses SPACE_USER / SPACE_PASS)
# --------------------------------
SPACE_USER = os.getenv("SPACE_USER", "user")  
SPACE_PASS = os.getenv("SPACE_PASS", "pass")

def check_login(u, p):
    ok = (u == SPACE_USER) and (p == SPACE_PASS)
    # if ok, nuke any prior session state
    new_st = new_state() if ok else gr.update()
    return (
        gr.update(visible=not ok),    
        gr.update(visible=ok),        
        ("" if ok else "Invalid username or password."), 
        new_st   
    )



# --- UI (defaults, no custom CSS/theme) -------------------------------
with gr.Blocks(fill_height=True) as ui:
    # ---- Login view (create these FIRST) ----
    with gr.Group(visible=True) as login_view:
        gr.Markdown("#### 🔒 Sign in")
        u = gr.Textbox(label="Username", autofocus=True)
        p = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Enter")
        login_msg = gr.Markdown("")

    # ---- App view (hidden until login succeeds) ----
    with gr.Group(visible=False) as app_view:
        gr.Markdown("### Spot Sepsis — Research Preview *(Not medical advice)*")
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(height=420, type="messages")
                msg = gr.Textbox(
                    placeholder="Describe the case (e.g., '2-year-old, HR 154, RR 36, SpO₂ 95%')",
                    lines=3
                )
                use_llm_chk = gr.Checkbox(value=USE_LLM_DEFAULT, label="Use OpenAI LLM (agent mode)")
                with gr.Row():
                    btn_run = gr.Button("Run")
                    btn_new = gr.Button("New chat")   # <-- add this
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
                "Errors related to timeout may occur on first use. Try again after ~1 minute if this happens."
                "</div>"
            )

        # state must exist before any wiring that references it
        state = gr.State(new_state())  # <-- move this up

        # --- callbacks ---
        def reset_all():
            # chat, state, info, paste, tips
            return [], new_state(), "", "", ""

        def on_user_send(history, text):
            history = history + [{"role": "user", "content": text}]
            return history, ""

        def on_bot_reply(history, st, stage, use_llm):
            st, reply = run_pipeline(st, history[-1]["content"], stage, use_llm=bool(use_llm))
            history = history + [{"role": "assistant", "content": reply}]
            info_json = json.dumps(st["sheet"], indent=2) if st.get("sheet") else ""
            return history, st, info_json, ""

        def on_merge(st, pasted):
            try:
                blob = json.loads(pasted)
            except Exception:
                return st, "Could not parse pasted JSON.", ""
            if st.get("sheet"):
                st["sheet"] = merge_sheet(
                    st["sheet"],
                    blob.get("features", {}).get("clinical", {}),
                    blob.get("features", {}).get("labs", {})
                )
            else:
                st["sheet"] = blob
            return st, "Merged.", json.dumps(st["sheet"], indent=2)

        # --- wiring (after widgets exist) ---
        login_btn.click(check_login, [u, p], [login_view, app_view, login_msg, state])

        msg.submit(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("auto"), use_llm_chk], [chat, state, info, msg]
        )
        btn_run.click(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("auto"), use_llm_chk], [chat, state, info, msg]
        )
        merge_btn.click(on_merge, [state, paste], [state, tips, info])

        # wire New chat after state exists
        btn_new.click(
            reset_all,
            inputs=None,
            outputs=[chat, state, info, paste, tips]
        )

# ---- Launch settings: Spaces vs local ---------------------------------
IS_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SYSTEM") == "spaces")
if IS_SPACES:
    ui.launch(ssr_mode=False)
else:
    ui.launch(server_name="127.0.0.1", server_port=7860)
