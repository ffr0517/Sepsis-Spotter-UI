import os, re, json, time, uuid
import requests
import gradio as gr
import logging, sys
from openai import OpenAI

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
You are Sepsis Spotter, a clinical intake and orchestration assistant for resource-limited settings (research preview; not a diagnosis).

# Core role
- Collect and validate inputs for Stage 1 (S1) and optionally Stage 2 (S2), then call the provided tools.
- Compose all user‑visible text yourself. The host app will not add prompts, disclaimers, or consent text for you.
- Keep messages concise, plain-language, and professional. No emojis.

# Tool discipline
- You have a single tool `sepsis_command` with actions: `ask`, `update_sheet`, or `call_api` (stage `S1` or `S2`).
- When you emit a tool call, you MUST also include a short user-facing message in the same turn explaining what you just did or need next.
- When the user provides enough labs for a validated S2 set and confirms SpO₂ is on room air, emit one tool call: {"action":"call_api","stage":"S2","features":{"labs":{…}}} (you may also include any newly parsed clinical in features.clinical). Do not send update_sheet first.
- Call at most one tool per turn.
# Data handling
- Never invent values. Parse/confirm from the user's words. If unsure, ask a single focused question.
- Convert years→months for age. Map sex: 1=male, 0=female.
- Gentle range checks (don’t block): HR 40–250; RR 10–120; SpO₂ 70–100; Temp 30–43 °C.
- When you extract multiple values from free text, emit them together via `update_sheet` before the next question.

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
- When essentials are present, emit call_api for S1 sending only the fields provided (omit unknowns).

# S1 contract (you must ensure readiness before calling)
Required keys present in `features.clinical` (omit unknowns entirely):
  age.months, sex (1/0), adm.recent (1/0), wfaz, cidysymp, not.alert (1/0), hr.all, rr.all, envhtemp, crt.long (1/0)
If any are missing, do NOT call S1; ask for what’s missing.

# S2 contract & validation
- Run S2 only after S1 (S1 meta-probs are computed by the backend from S1 response and stored in `sheet.features.clinical`).
- S2 validated sets you may proceed with directly:
  Set A: CRP (mg/L), TNFR1 (pg/ml), suPAR (ng/ml), SpO₂ on room air (%).
  Set B: CRP (mg/L), CXCl10 (pg/ml), IL-6 (pg/ml), SpO₂ on room air (%).
- If labs do not match Set A or B but a larger panel exists (≈6+ markers), you may proceed as `full_lab_panel`.
- If none of these apply, warn the user that the combination is NOT VALIDATED and ask for explicit confirmation before calling S2. If the host state `awaiting_unvalidated_s2` is true, they already confirmed.

# Results & disclaimers
- When the backend returns S1/S2 results (now present in the conversation context `sheet`), you must summarize the outcome and include a clear clinical-decision-support disclaimer in your own words.
- You decide how to phrase results and any next-step suggestions (e.g., propose S2 after S1=Other), keeping things short.

# UI contract
- The host provides you with a JSON `sheet` structure in the user message context. Read from it; write to it only through the tool.
- After the host runs your tool call, it will update `sheet` and call you again with the new `sheet`.

# Field dictionary (for your reference; do not display raw keys to the user)
- age.months (months); sex (1=male,0=female); bgcombyn (comorbidity 1/0); adm.recent (overnight hospitalisation within 6 months 1/0); wfaz (WFA Z);
- waste (WFLZ<-2 1/0); stunt (LAZ<-2 1/0); cidysymp (illness duration, days); prior.care (1/0); travel.time.bin (≤1h 1/0);
- diarrhoeal (1/0); pneumo (1/0); sev.pneumo (1/0); ensapro (prostration 1/0); vomit.all (1/0); seiz (1/0); pfacleth (lethargy 1/0);
- not.alert (AVPU<A 1/0); danger.sign (IMCI danger sign 1/0); hr.all (bpm); rr.all (/min); oxy.ra (% on room air);
- envhtemp (°C); crt.long (>2s 1/0); parenteral_screen (1/0); SIRS_num (0–4);
- Labs include: CRP, TNFR1, supar, CXCl10, IL6, IL10, IL1ra, IL8, PROC, ANG1, ANG2, CHI3L, STREM1, VEGFR1, lblac, lbglu, enescbchb1.
"""

# ------------------------------
# Tool schema exposed to the LLM
# ------------------------------
TOOL_SPEC = [{
    "type": "function",
    "name": "sepsis_command",
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
    return merge_sheet(
        sheet,
        (feats or {}).get("clinical", {}) or {},
        (feats or {}).get("labs", {}) or {},
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
    r = requests.post(API_S1, json={"features": payload}, timeout=30)
    r.raise_for_status()
    return r.json()


def call_s2(features, apply_calibration=True, allow_heavy_impute=False):
    payload = {"features": features, "apply_calibration": bool(apply_calibration)}
    if allow_heavy_impute:
        payload["allow_heavy_impute"] = True
    r = requests.post(API_S2, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# ------------------------------
# Validation helpers (host-side; we never craft user text here)
# ------------------------------

def missing_for_s1(clinical: dict):
    return [k for k in S1_REQUIRED_MIN if k not in (clinical or {}) or (clinical or {}).get(k) in (None, "")]


def validated_set_name(features: dict) -> str | None:
    f = features or {}

    def provided(k):
        if k not in f:
            return False
        v = f[k]
        if v is None:
            return False
        if isinstance(v, str) and v.strip() == "":
            return False
        if v == 0 or v == 0.0:
            return False
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
    """Single LLM call that may return assistant text plus (at most one) tool call."""
    context = {"sheet": sheet}
    input_items = [
        {"type": "message", "role": "system", "content": [{"type": "input_text", "text": AGENT_SYSTEM}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"CONTEXT\n{json.dumps(context, indent=2)}\n\nUSER\n{(user_text or '').strip()}"}]},
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
        text={"verbosity": "low"},
        reasoning={"effort": "high"},   # <— high‑effort reasoning enabled
        parallel_tool_calls=False,
        max_tool_calls=1,
        store=False,
    )

    say = ""
    cmd = None
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

def run_pipeline(state, user_text, stage="auto", use_llm=True):
    state = state or {"sheet": None, "conv_id": None}
    sheet = state.get("sheet") or new_sheet()

    # Legacy extraction if LLM disabled/unavailable
    if not use_llm or len(os.getenv("OPENAI_API_KEY", "").strip()) < 20 or PARSER_MODE != "llm_only":
        clin_new, labs_new, _ = extract_features(user_text or "")
        if clin_new or labs_new:
            sheet = merge_sheet(sheet, clin_new, labs_new)
            state["sheet"] = sheet
        # In legacy mode we do no extra messaging here; just echo parsed and leave dialogue to the user
        return state, "Parsed what I could. Please continue."

    # Agent pass #1
    say, cmd = agent_call(user_text=user_text, sheet=sheet, conv_id=state.get("conv_id"))

    # Apply at most one tool call (no host-authored phrasing)
    if cmd:
        action = (cmd or {}).get("action")
        stage_req = (cmd or {}).get("stage") or "auto"
        feats = (cmd or {}).get("features") or {}

        if action == "update_sheet":
            sheet = merge_features(sheet, feats)
            state["sheet"] = sheet
            # Let the LLM do any follow-up phrasing
            say2 = agent_followup(sheet, note="update_sheet applied")
            return state, (say or "") + ("\n\n" + say2 if say2 else "")

        if action == "ask":
            # LLM already produced the question in `say`
            return state, (say or "Could you share one detail?")

        if action == "call_api":
            # Merge any features included with the call
            sheet = merge_features(sheet, feats)
            state["sheet"] = sheet

            # Decide stage (host enforces contracts silently, then asks LLM to explain)
            stage_eff = stage_req
            if stage_eff == "auto":
                stage_eff = "S2" if sheet.get("features", {}).get("labs") else "S1"

            try:
                if stage_eff == "S1":
                    missing = missing_for_s1(sheet.get("features", {}).get("clinical", {}))
                    if missing:
                        # Tell the LLM about the problem; it will phrase the ask
                        say2 = agent_followup(sheet, note=f"S1 missing fields: {missing}")
                        return state, (say or "") + ("\n\n" + say2 if say2 else "")
                    s1 = call_s1(sheet["features"]["clinical"])
                    sheet["s1"] = s1
                    # Compute & store meta-probs for S2 if provided by backend
                    def _as_float(x):
                        try: return float(x)
                        except: return None
                    v1p = _as_float(((s1 or {}).get("v1") or {}).get("prob"))
                    v2p = _as_float(((s1 or {}).get("v2") or {}).get("prob"))
                    if v1p is not None:
                        sheet["features"]["clinical"]["v1_pred_Severe"] = v1p
                        sheet["features"]["clinical"]["v1_pred_Other"] = 1.0 - v1p
                    if v2p is not None:
                        sheet["features"]["clinical"]["v2_pred_NOTSevere"] = v2p
                        sheet["features"]["clinical"]["v2_pred_Other"] = 1.0 - v2p
                    state["sheet"] = sheet
                    say2 = agent_followup(sheet, note="S1 result ready")
                    return state, (say or "") + ("\n\n" + say2 if say2 else "")

                elif stage_eff == "S2":
                    # Ensure validated or explicit confirmation
                    merged = {**sheet.get("features", {}).get("clinical", {}), **sheet.get("features", {}).get("labs", {})}
                    vname = validated_set_name(merged)
                    if vname is None and not state.get("awaiting_unvalidated_s2"):
                        # Flag for the LLM to ask for confirmation in its own words
                        state["awaiting_unvalidated_s2"] = True
                        say2 = agent_followup(sheet, note="S2 set not validated; request explicit confirmation or alternate set")
                        return state, (say or "") + ("\n\n" + say2 if say2 else "")

                    s2 = call_s2(merged, apply_calibration=True)
                    sheet["s2"] = s2
                    state["sheet"] = sheet
                    state["awaiting_unvalidated_s2"] = False
                    say2 = agent_followup(sheet, note="S2 result ready")
                    return state, (say or "") + ("\n\n" + say2 if say2 else "")

                else:
                    say2 = agent_followup(sheet, note=f"Unknown stage: {stage_eff}")
                    return state, (say or "") + ("\n\n" + say2 if say2 else "")

            except requests.RequestException as e:
                err = f"API request error: {e}"
                log.exception(err)
                say2 = agent_followup(sheet, note=err)
                return state, (say or "") + ("\n\n" + say2 if say2 else "")
            except Exception as e:
                err = f"Pipeline error: {e}"
                log.exception(err)
                say2 = agent_followup(sheet, note=err)
                return state, (say or "") + ("\n\n" + say2 if say2 else "")

    # If no tool call, just return the LLM’s prose
    return state, (say or "Okay.")

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
                msg = gr.Textbox(placeholder="Describe the case…", lines=3)
                use_llm_chk = gr.Checkbox(value=USE_LLM_DEFAULT, label="Use OpenAI LLM (agent mode)")
                with gr.Row():
                    btn_run = gr.Button("Run")
                    btn_new = gr.Button("New chat")
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
                    "Timeouts may occur on first use if the backend is cold. Try again shortly."
                    "</div>"
                )

        state = gr.State(new_state())

        def reset_all():
            return [], new_state(), "", "", ""

        def on_user_send(history, text):
            history = history + [{"role": "user", "content": text}]
            return history, ""

        def on_bot_reply(history, st, stage, use_llm):
            st, reply = run_pipeline(st, history[-1]["content"], stage, use_llm=bool(use_llm))
            history = history + [{"role": "assistant", "content": reply}]
            info_json = json.dumps(st.get("sheet", {}), indent=2)
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

        login_btn.click(check_login, [u, p], [login_view, app_view, login_msg, state])
        msg.submit(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("auto"), use_llm_chk], [chat, state, info, msg]
        )
        btn_run.click(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("auto"), use_llm_chk], [chat, state, info, msg]
        )
        merge_btn.click(on_merge, [state, paste], [state, tips, info])
        btn_new.click(reset_all, inputs=None, outputs=[chat, state, info, paste, tips])

IS_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SYSTEM") == "spaces")
if IS_SPACES:
    ui.launch(ssr_mode=False)
else:
    ui.launch(server_name="127.0.0.1", server_port=7860)

