import os, re, json, time, requests, gradio as gr

import logging, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("sepsis-agent")

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))


DEBUG_AGENT = bool(int(os.getenv("DEBUG_AGENT", "0")))

USE_LLM_DEFAULT = True  # default for the UI checkbox

# ------------------------------
# Agent system prompt (LLM is the orchestrator)
# ------------------------------
AGENT_SYSTEM = """
You are SepsisAgent (research preview; not medical advice).
- Greet and speak naturally (brief, friendly).
- Collect only the fields required for S1/S2; ask ONE question at a time when missing.
- S1 needs: age.months, sex (0 male, 1 female), hr.all, rr.all, oxy.ra.
- Prefer S2 if labs {CRP, PCT, Lactate, WBC, Neutrophils, Platelets} are present.
- Convert years→months; map sex male/boy→0, female/girl→1; do not invent values.
- When you want to take an action, CALL the tool `sepsis_command` with action + message + features (and optional stage).
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
    if re.search(r"\bmale\b|\bboy\b", text, re.I): clinical["sex"] = 0
    if re.search(r"\bfemale\b|\bgirl\b", text, re.I): clinical["sex"] = 1

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
    payload = {"features": clinical}
    r = requests.post(API_S1, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def call_s2(features):
    payload = {"features": features}
    r = requests.post(API_S2, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

# --------------------------------
# Agent helpers (LLM orchestrator)
# --------------------------------
def _get_llm_model():
    m = os.getenv("LLM_MODEL_ID", "").strip()
    if not m or "/" in m:
        return "gpt-4o-mini"
    return m

def agent_step(user_text: str, sheet: dict | None, conv_id: str | None):
    """
    Ask the model to chat naturally AND optionally emit a structured tool call.
    Returns: (say_text, cmd_dict_or_None, new_conv_id)
    """
    sheet = sheet or new_sheet()
    context = {"sheet": sheet}

    # Responses API expects items, not chat-style role/content pairs; use a system message + a user message.
    input_msgs = [
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

    # Tools schema for Responses API (function tool)
    tools = [
        {
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
                            "clinical": {"type": "object", "additionalProperties": True},
                            "labs": {"type": "object", "additionalProperties": True},
                        },
                    },
                    "stage": {"type": "string", "enum": ["auto", "S1", "S2"]},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        }
    ]

    resp = client.responses.create(
        model=_get_llm_model(),
        input=input_msgs,           # <-- use input_msgs here
        tools=tools,
        conversation=conv_id,       # None on first turn; OpenAI creates one
        temperature=0,
    )

    say = ""
    cmd = None
    new_conv_id = getattr(resp, "conversation", {}).get("id", None)

    for item in (resp.output or []):
        if item.get("type") == "message" and item.get("role") == "assistant":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    say += c.get("text", "")

        if item.get("type") == "tool_call" and item.get("name") == "sepsis_command":
            try:
                cmd = json.loads(item.get("arguments") or "{}")
            except Exception:
                cmd = None

    if DEBUG_AGENT:
        try:
            log.info("[RESPONSES RAW]\n%s", resp.model_dump_json(indent=2))
        except Exception:
            log.info("[RESPONSES RAW] %s", resp)
        log.info("[RESPONSES SAY] %s", (say or "").strip())
        log.info("[RESPONSES CMD] %s", json.dumps(cmd, indent=2) if cmd else "(none)")

    return (say.strip() or None), cmd, (new_conv_id or conv_id)

# --------------------------------
# Orchestration (Agent-first when toggle ON)
# --------------------------------
DEBUG_AGENT = bool(int(os.getenv("DEBUG_AGENT", "0")))

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
            summary = f"**S1 decision:** {s1.get('s1_decision')}\n\n**Current Info Sheet:**\n```json\n{json.dumps(sheet, indent=2)}\n```"
            return state, summary
        elif stage == "S2":
            features = {**sheet["features"]["clinical"], **sheet["features"]["labs"]}
            s2 = call_s2(features)
            sheet["s2"] = s2
            state["sheet"] = sheet
            summary = f"**S2 decision:** {s2.get('s2_decision')}\n\n**Current Info Sheet:**\n```json\n{json.dumps(sheet, indent=2)}\n```"
            return state, summary
        else:
            return state, "Unknown stage."
    except Exception as e:
        return state, f"Error calling API: {e}"
    
def run_pipeline(state, user_text, stage="auto", use_llm=True):
    state.setdefault("sheet", None)
    state.setdefault("conv_id", None)

    if use_llm:
        say, cmd, conv_id = agent_step(user_text, state["sheet"], state["conv_id"])
        state["conv_id"] = conv_id

        # If the model just talks (no tool call), show its message
        if not cmd:
            return state, (say or "…")

        action = cmd.get("action")

        if action == "update_sheet":
            state["sheet"] = merge_features(state.get("sheet") or new_sheet(), cmd.get("features"))
            return state, (cmd.get("message") or say or "")

        if action == "ask":
            return state, (cmd.get("message") or say or "")

        if action == "call_api":
            fs = cmd.get("features") or {}
            stage_req = cmd.get("stage") or "auto"
            sheet = merge_features(state.get("sheet") or new_sheet(), fs)
            if stage_req == "auto":
                stage_req = "S2" if sheet["features"]["labs"] else "S1"
            try:
                if stage_req == "S1":
                    s1 = call_s1(sheet["features"]["clinical"])
                    sheet["s1"] = s1
                    state["sheet"] = sheet
                    return state, f"{cmd.get('message') or say or ''}\n\n**S1 decision:** {s1.get('s1_decision')}\n\n```json\n{json.dumps(sheet, indent=2)}\n```"
                else:
                    features = {**sheet["features"]["clinical"], **sheet["features"]["labs"]}
                    s2 = call_s2(features)
                    sheet["s2"] = s2
                    state["sheet"] = sheet
                    return state, f"{cmd.get('message') or say or ''}\n\n**S2 decision:** {s2.get('s2_decision')}\n\n```json\n{json.dumps(sheet, indent=2)}\n```"
            except Exception as e:
                return state, f"{cmd.get('message') or say or ''}\n\nError calling API: {e}"

        # Unknown action → just show the assistant text
        return state, (say or "")

    # toggle off → legacy regex path
    return run_pipeline_legacy(state, user_text, stage)

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

with gr.Blocks(fill_height=True) as ui:
    # ---- Login view ----
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
                # ✅ LLM orchestrator toggle
                use_llm_chk = gr.Checkbox(
                    value=USE_LLM_DEFAULT,
                    label="Use OpenAI LLM (agent mode)"
                )
                with gr.Row():
                    btn_s1 = gr.Button("Run S1")
                    btn_s2 = gr.Button("Run S2 (with labs)")
                    btn_auto = gr.Button("Auto")
            with gr.Column(scale=2):
                info = gr.Textbox(label="Current Info Sheet (JSON)", lines=22)
                paste = gr.Textbox(label="Paste an Info Sheet to restore/merge", lines=6)
                merge_btn = gr.Button("Merge")
                tips = gr.Markdown("")

        state = gr.State({"sheet": None, "conv_id": None})

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

        # Wire chat + buttons (pass checkbox through)
        msg.submit(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("auto"), use_llm_chk], [chat, state, info, msg]
        )
        btn_s1.click(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("S1"), use_llm_chk], [chat, state, info, msg]
        )
        btn_s2.click(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("S2"), use_llm_chk], [chat, state, info, msg]
        )
        btn_auto.click(on_user_send, [chat, msg], [chat, msg]).then(
            on_bot_reply, [chat, state, gr.State("auto"), use_llm_chk], [chat, state, info, msg]
        )
        merge_btn.click(on_merge, [state, paste], [state, tips, info])

    # Wire the login button
    login_btn.click(check_login, [u, p], [login_view, app_view, login_msg])

# ---- Launch settings: Spaces vs local ---------------------------------
IS_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SYSTEM") == "spaces")
if IS_SPACES:
    # On Spaces: let proxy choose host/port; disable SSR to avoid locale/i18n issues.
    ui.launch(ssr_mode=False)
else:
    ui.launch(server_name="127.0.0.1", server_port=7860)
