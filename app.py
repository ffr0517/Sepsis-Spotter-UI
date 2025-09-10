import os, re, json, time, requests, gradio as gr

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
You are Sepsis Spotter, a clinical intake & orchestration assistant (research preview; not a diagnosis).

FIRST TURN ONLY
- Start with: “I am using <MODEL_NAME>.” (or “an AI language model” if unknown) + “This is clinical decision support, not a diagnosis.”
- Do not restate model identity or the disclaimer again.

STYLE & BOUNDARIES
- Be brief, clear, and clinical. No emojis.
- Never show internal field names (e.g., “rr.all”). Use plain language (“breathing rate”).
- One tool call per turn via `sepsis_command`: `ask`, `update_sheet`, or `call_api`.

INTAKE & NORMALIZATION
- Parse and normalize units:
  - Age: years/days/weeks → months (e.g., “one and a half” → 18 mo; 60 weeks → 13.8 mo).
  - Sex: map internally (1=male, 0=female) but never expose mapping.
- If user provides values in free text, emit `update_sheet` with all confidently parsed values from that message.
- Key plausibility ranges (gentle checks): HR 40–250 bpm, RR 10–120 /min, SpO₂ 70–100 %.

MISSING-INFO LOGIC (exactly one ask at a time)
- If any of the following are missing or unclear, `ask` for the single highest-impact one:
  1) Breathing rate (breaths/min),
  2) Oxygen saturation on room air (%)—if user says “oxygen looks fine”, ask for the exact % and confirm room air,
  3) Alertness (e.g., “alert” vs “not alert”).
- Otherwise, proceed.

MODEL SELECTION
- S1 only (clinical-only). If labs are mentioned, reply: “The lab-based model isn’t available; I’ll use the clinical model (S1).” Do not attempt S2.

PRE-FLIGHT CONFIRMATION (before any `call_api`)
- Present one concise line with the values you will send and any assumptions, e.g.:
  “Sending: age 18 mo, female, HR 132, RR  — , SpO₂ — % on room air, alert — . Assumptions: none. Shall I run S1?”
- Do not call the API until the user confirms.
- Unknowns are allowed at call time but must be explicitly labeled as “unknown (placeholder)” in the confirmation.

CALL CONTRACT (internal; don’t expose keys)
- When calling S1, include the full clinical feature set.
- Unknowns: binary → 0, continuous → 0.0.
- Never expose internal key names in chat.

OUTPUT
- Keep ≤4 sentences.
- State the model used (S1) and a brief, actionable summary.
- Always end results with: “This is clinical decision support, not a diagnosis.”

ERRORS
- If the endpoint errors or times out: say it failed, don’t cache or invent results, and return to `ask` for the most likely culprit next turn.
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
API_S1 = os.getenv("SEPSIS_API_URL_S1", "https://sepsis-spotter-beta.onrender.com/s1_infer")
API_S2 = os.getenv("SEPSIS_API_URL_S2", "https://sepsis-spotter-beta.onrender.com/s2_infer")

# Required fields for S1 (adjust to your exact schema)
REQUIRED_S1 = ["age.months","sex","hr.all","rr.all","oxy.ra"]
OPTIONAL_S1 = [
    "wfaz","SIRS_num","crt.long","prior.care","danger.sign","not.alert",
    "urti","lrti","diarrhoeal","envhtemp","parenteral_screen"
]
LAB_KEYS = ["CRP","PCT","Lactate","WBC","Neutrophils","Platelets"]  # add/remove as needed

# --------------------------------
# Simple rule-based extractor (robust + free) for legacy path
# --------------------------------
def extract_features(text: str):
    clinical, labs, notes = {}, {}, []

    # age: years or months
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:years?|yrs?|y)\b", text, re.I)
    if m: clinical["age.months"] = float(m.group(1)) * 12
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mos?|mo)\b", text, re.I)
    if m: clinical["age.months"] = float(m.group(1))

    # sex
    if re.search(r"\bmale\b|\bboy\b", text, re.I): clinical["sex"] = 1
    if re.search(r"\bfemale\b|\bgirl\b", text, re.I): clinical["sex"] = 0

    # vitals
    m = re.search(r"\bHR[:\s]*([0-9]{2,3})\b", text, re.I)
    if m: clinical["hr.all"] = int(m.group(1))
    m = re.search(r"\bRR[:\s]*([0-9]{1,3})\b", text, re.I)
    if m: clinical["rr.all"] = int(m.group(1))
    m = re.search(r"(?:SpO2|SpO₂|sats?|oxygen|sat)[:\s]*([0-9]{2,3})\b", text, re.I)
    if m: clinical["oxy.ra"] = int(m.group(1))

    # common flags
    if re.search(r"\bdanger sign(s)?\b", text, re.I): clinical["danger.sign"] = 1
    if re.search(r"\bnot alert\b|\bletharg(y|ic)\b|\bdrows(y|iness)\b", text, re.I): clinical["not.alert"] = 1
    if re.search(r"\bURTI\b|\bupper resp", text, re.I): clinical["urti"] = 1
    if re.search(r"\bLRTI\b|\blower resp|\bpneumonia\b", text, re.I): clinical["lrti"] = 1
    if re.search(r"\bdiarrh", text, re.I): clinical["diarrhoeal"] = 1
    m = re.search(r"\bcrt[:\s]*([0-9](?:\.[0-9])?)\b", text, re.I)
    if m: clinical["crt.long"] = 1 if float(m.group(1)) >= 3 else 0

    # labs (simple)
    for k in LAB_KEYS:
        m = re.search(fr"\b{k}\b[:\s]*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        if m: labs[k] = float(m.group(1))

    return clinical, labs, notes

def validate_complete(clinical: dict):
    missing = [k for k in REQUIRED_S1 if k not in clinical]
    warnings = []
    # basic range checks (tune to your data)
    if "oxy.ra" in clinical and not (50 <= int(clinical["oxy.ra"]) <= 100):
        warnings.append("SpO2 seems out of range.")
    if "hr.all" in clinical and not (40 <= int(clinical["hr.all"]) <= 250):
        warnings.append("HR seems out of range.")
    if "rr.all" in clinical and not (10 <= int(clinical["rr.all"]) <= 120):
        warnings.append("RR seems out of range.")
    if "age.months" in clinical and not (0 <= float(clinical["age.months"]) <= 180):
        warnings.append("Age (months) seems out of range.")
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

# --------------------------------
# Model calls
# --------------------------------
def call_s1(clinical):
    try:
        r = requests.post(API_S1, json={"features": clinical}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"S1 endpoint unavailable or returned an error: {e}") from e

def call_s2(features):
    try:
        r = requests.post(API_S2, json={"features": features}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"S2 endpoint unavailable or returned an error: {e}") from e

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
    if new_conv:
        state["conv_id"] = new_conv

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

    if action == "ask":
        # Just pass the model's concise question through
        state["sheet"] = sheet
        return state, (message or "Could you clarify one detail?")

    if action == "update_sheet":
        feats = (cmd or {}).get("features") or {}
        sheet = merge_features(sheet, feats)
        state["sheet"] = sheet

        # If model didn't supply a next question, try to prompt for the next missing S1 field
        missing, warnings = validate_complete(sheet["features"]["clinical"])
        if stage_hint == "S1" or stage_hint == "auto":
            if missing:
                next_required = missing[0]
                nxt = f"{message}\n\nCould you provide **{next_required}**?"
            else:
                nxt = message or "Thanks — I have the essentials."
        else:
            nxt = message or "Got it."

        if warnings:
            nxt += "\n\n⚠️ " + " ".join(warnings)

        return state, nxt

    if action == "call_api":
        stage = (cmd or {}).get("stage") or "auto"
        # Merge any features the tool call included
        feats = (cmd or {}).get("features") or {}
        sheet = merge_features(sheet, feats)
        state["sheet"] = sheet

        # Choose stage
        if stage == "auto":
            stage = "S2" if sheet["features"]["labs"] else "S1"

        # Validate for S1
        if stage == "S1":
            missing, warnings = validate_complete(sheet["features"]["clinical"])
            if missing:
                return state, "Missing required fields for S1: " + ", ".join(missing) + "."

            try:
                s1 = call_s1(sheet["features"]["clinical"])
                sheet["s1"] = s1
                state["sheet"] = sheet
                reply = (message or "Running S1 now.") + f"\n\n**S1 decision:** {s1.get('s1_decision')}"
                return state, reply
            except Exception as e:
                return state, f"Error calling S1 API: {e}"

        elif stage == "S2":
            try:
                features = {**sheet["features"]["clinical"], **sheet["features"]["labs"]}
                s2 = call_s2(features)
                sheet["s2"] = s2
                state["sheet"] = sheet
                reply = (message or "Running S2 now.") + f"\n\n**S2 decision:** {s2.get('s2_decision')}"
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
            state["sheet"] = sheet
            # Chat reply: decision only (no JSON dump)
            summary = f"**S1 decision:** {s1.get('s1_decision')}"
            return state, summary
        elif stage == "S2":
            features = {**sheet["features"]["clinical"], **sheet["features"]["labs"]}
            s2 = call_s2(features)
            sheet["s2"] = s2
            state["sheet"] = sheet
            summary = f"**S2 decision:** {s2.get('s2_decision')}"
            return state, summary
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
        # conversation=conv_id,  # optional; omit unless you really want server-side memory
    )

    say = ""
    cmds = []
    new_conv_id = getattr(resp, "conversation", None).id if getattr(resp, "conversation", None) else None

    for item in (resp.output or []):
        # Assistant text (if any)
        if getattr(item, "type", "") == "message" and getattr(item, "role", "") == "assistant":
            for c in (getattr(item, "content", []) or []):
                if getattr(c, "type", "") == "output_text":
                    say += (getattr(c, "text", "") or "")

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

    return (say.strip() or None), cmds, (new_conv_id or conv_id)


# --------------------------------
# In-app Login + Gradio UI (uses SPACE_USER / SPACE_PASS)
# --------------------------------
SPACE_USER = os.getenv("SPACE_USER", "user")   # set in HF Space → Settings → Variables & secrets
SPACE_PASS = os.getenv("SPACE_PASS", "pass")

def check_login(u, p):
    ok = (u == SPACE_USER) and (p == SPACE_PASS)
    # Hide login, show app if ok; otherwise show error
    return (
        gr.update(visible=not ok),   # login_view
        gr.update(visible=ok),       # app_view
        ("" if ok else "Invalid username or password.")
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
                    btn_run = gr.Button("Run")  # single button, always 'auto'
            with gr.Column(scale=2):
                info = gr.Textbox(label="Current Info Sheet (JSON)", lines=22)
                paste = gr.Textbox(label="Paste an Info Sheet to restore/merge", lines=6)
                merge_btn = gr.Button("Merge")
                tips = gr.Markdown("")

        state = gr.State({"sheet": None, "conv_id": None})

        # --- callbacks ---
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
        login_btn.click(check_login, [u, p], [login_view, app_view, login_msg])

        msg.submit(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("auto"), use_llm_chk], [chat, state, info, msg]
        )
        btn_run.click(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("auto"), use_llm_chk], [chat, state, info, msg]
        )
        merge_btn.click(on_merge, [state, paste], [state, tips, info])

# ---- Launch settings: Spaces vs local ---------------------------------
IS_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SYSTEM") == "spaces")
if IS_SPACES:
    ui.launch(ssr_mode=False)
else:
    ui.launch(server_name="127.0.0.1", server_port=7860)
