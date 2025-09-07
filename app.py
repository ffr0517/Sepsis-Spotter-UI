import os, re, json, time, requests, gradio as gr

import logging, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("sepsis-agent")

DEBUG_AGENT = bool(int(os.getenv("DEBUG_AGENT", "0")))

# ---- Optional tiny LLM for parsing/orchestration (CPU/Spaces) ----
USE_LLM_DEFAULT = True  # default for the UI checkbox
MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")  # tiny & fast on CPU
_llm_pipe = None

def _load_tiny_llm():
    """
    Lazy-load a very small instruct model suitable for CPU (Spaces Basic).
    """
    global _llm_pipe
    if _llm_pipe is not None:
        return _llm_pipe
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,       # safe on CPU
            low_cpu_mem_usage=True,
        )
        _llm_pipe = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            max_new_tokens=256,             # a bit more room for the agent command
            do_sample=False,                # deterministic; no temperature needed
        )
        # Pre-warm a touch so first real call is faster
        try:
            _llm_pipe("{}", max_new_tokens=1)
        except Exception:
            pass
        return _llm_pipe
    except Exception as e:
        print("LLM load failed, falling back to regex:", e)
        _llm_pipe = None
        return None

# ------------------------------
# Agent system prompt (LLM is the orchestrator)
# ------------------------------
AGENT_SYSTEM = """
You are SepsisAgent in a research preview (not medical advice).
Your job: converse naturally, extract required fields, ask for missing info (one question at a time), and call S1/S2 when ready.

API:
- S1 requires: age.months, sex (0 male, 1 female), hr.all, rr.all, oxy.ra
- S2 uses clinical + any labs (CRP, PCT, Lactate, WBC, Neutrophils, Platelets). Prefer S2 if labs exist and clinical is sufficient.

Conventions: convert years→months; map sex male/boy→0, female/girl→1; do not invent values.

OUTPUT: return EXACTLY ONE JSON object between <<<JSON> and </JSON>>> with one of:
1) {"action":"ask","message":"<your own brief question>"} 
2) {"action":"update_sheet","features":{"clinical":{...},"labs":{...}},"message":"<your own brief acknowledgement>"} 
3) {"action":"call_api","stage":"auto"|"S1"|"S2","features":{"clinical":{...},"labs":{...}},"message":"<your own brief status>"}
No text outside the JSON tags.
"""


# ------------------------------
# Legacy extractor prompt (still available for toggle OFF path)
# ------------------------------
SYSTEM_PROMPT = (
    "You are a clinical intake assistant. Extract only structured fields needed for a sepsis risk model.\n"
    "Return STRICT JSON with keys:\n"
    "{\n"
    '  \"clinical\": {\n'
    '    \"age.months\": <number>, \"sex\": <0 for male, 1 for female>, \"hr.all\": <int>,\n'
    '    \"rr.all\": <int>, \"oxy.ra\": <int>\n'
    "  },\n"
    '  \"labs\": { \"CRP\": <number>, \"PCT\": <number>, \"Lactate\": <number>, \"WBC\": <number>, \"Neutrophils\": <number>, \"Platelets\": <number> }\n'
    "}\n"
    "Rules:\n"
    "- Convert years to months.\n"
    "- For sex, map male/boy→0, female/girl→1.\n"
    "- If a value is missing, omit the key. Do not invent values.\n"
    "- Output ONLY JSON. No commentary.\n"
)

# Safe call with timeout + JSON extraction (legacy helper)
def llm_extract_to_dict(user_text: str, timeout_s: int = 15) -> dict:
    """
    Run a small LLM to extract structured fields. If anything goes wrong, return {} so regex can fill gaps.
    """
    pipe = _load_tiny_llm()
    if pipe is None:
        return {}

    import concurrent.futures

    prompt = f"{SYSTEM_PROMPT}\n\nUser text:\n{(user_text or '').strip()}\n\nJSON:"

    def _call():
        out = pipe(prompt)[0]["generated_text"]
        m = re.search(r"\{[\s\S]*\}\s*$", out)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_call)
            return fut.result(timeout=timeout_s)
    except Exception:
        return {}

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
def _coerce_json(s: str) -> dict:
    if not s:
        return {}
    # Prefer content between our explicit tags
    m = re.search(r"<<<JSON>\s*(\{[\s\S]*?\})\s*</JSON>>>", s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Fallback: last {...} anywhere
    candidates = re.findall(r"\{[\s\S]*?\}", s)
    for chunk in reversed(candidates):
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return {}

def agent_step(user_text: str, sheet: dict | None):
    """
    Build a minimal context (current sheet + user turn), ask the LLM for ONE command JSON.
    """
    pipe = _load_tiny_llm()
    if pipe is None:
        return {"action": "fallback"}  # triggers legacy path

    sheet = sheet or new_sheet()
    context = {
        "sheet": sheet,
        "note": "Reply with exactly one JSON command per the contract."
    }
    prompt = (
        AGENT_SYSTEM
        + "\n\n--- CONTEXT ---\n"
        + json.dumps(context, indent=2)
        + "\n\n--- USER ---\n"
        + (user_text or "").strip()
        + "\n\n<<<JSON>\n"  # open the JSON block so the model fills it
    )

    out = pipe(prompt, max_new_tokens=240, do_sample=False)[0]["generated_text"]
    if DEBUG_AGENT:
        log.info("[AGENT RAW OUTPUT]\n%s\n[/AGENT RAW OUTPUT]", out)

    cmd = _coerce_json(out)

    if DEBUG_AGENT:
        log.info("[AGENT PARSED CMD]\n%s", json.dumps(cmd, indent=2) if cmd else "(none)")

    return cmd

def merge_features(sheet, feats):
    return merge_sheet(
        sheet,
        (feats or {}).get("clinical", {}),
        (feats or {}).get("labs", {})
    )

# --------------------------------
# Legacy pipeline (used when toggle OFF or agent fallback)
# --------------------------------
def run_pipeline_legacy(state, user_text, stage="auto"):
    # 1) Try LLM extraction (legacy) then regex fallback
    blob = llm_extract_to_dict(user_text or "")
    clin_new = (blob.get("clinical") or {}) if isinstance(blob, dict) else {}

    rx_clin, rx_labs, _ = extract_features(user_text or "")
    for k, v in rx_clin.items():
        if k not in clin_new:
            clin_new[k] = v
    labs_new = (blob.get("labs") or {}) if isinstance(blob, dict) else {}
    for k, v in rx_labs.items():
        if k not in labs_new:
            labs_new[k] = v

    sheet = state.get("sheet") or new_sheet()
    sheet = merge_sheet(sheet, clin_new, labs_new)

    # Decide stage
    if stage == "auto":
        stage = "S2" if sheet["features"]["labs"] else "S1"

    # Validate S1
    missing, warns = validate_complete(sheet["features"]["clinical"])
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

# --------------------------------
# Orchestration (Agent-first when toggle ON)
# --------------------------------
DEBUG_AGENT = bool(int(os.getenv("DEBUG_AGENT", "0")))

def run_pipeline(state, user_text, stage="auto", use_llm=True):
    state.setdefault("sheet", None)
    if use_llm:
        cmd = agent_step(user_text, state["sheet"])
        if not cmd:
            if DEBUG_AGENT:
                return state, "(agent output invalid JSON; check logs)"
            # In prod, you can silently retry agent_step(...) once; otherwise return a minimal neutral token:
            return state, "…"  # neutral placeholder; no guidance text from you

        action = cmd.get("action")
        if action == "update_sheet":
            state["sheet"] = merge_features(state.get("sheet") or new_sheet(), cmd.get("features"))
            return state, (cmd.get("message") or "")
        if action == "ask":
            return state, (cmd.get("message") or "")
        if action == "call_api":
            fs = cmd.get("features") or {}
            stage_req = cmd.get("stage") or "auto"
            sheet = merge_features(state.get("sheet") or new_sheet(), fs)
            if stage_req == "auto":
                stage_req = "S2" if sheet["features"]["labs"] else "S1"
            # (optional) guard on S1 minimums; otherwise let LLM fully drive
            try:
                if stage_req == "S1":
                    s1 = call_s1(sheet["features"]["clinical"])
                    sheet["s1"] = s1
                    state["sheet"] = sheet
                    return state, f"{cmd.get('message','')}\n\n**S1 decision:** {s1.get('s1_decision')}\n\n```json\n{json.dumps(sheet, indent=2)}\n```"
                else:
                    features = {**sheet["features"]["clinical"], **sheet["features"]["labs"]}
                    s2 = call_s2(features)
                    sheet["s2"] = s2
                    state["sheet"] = sheet
                    return state, f"{cmd.get('message','')}\n\n**S2 decision:** {s2.get('s2_decision')}\n\n```json\n{json.dumps(sheet, indent=2)}\n```"
            except Exception as e:
                return state, f"Error calling API: {e}"
        return state, ""  # unknown action → no extra copy
    # toggle off → legacy
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
                chat = gr.Chatbot(height=420, type="tuples")
                msg = gr.Textbox(
                    placeholder="Describe the case (e.g., '2-year-old, HR 154, RR 36, SpO₂ 95%')",
                    lines=3
                )
                # ✅ LLM orchestrator toggle
                use_llm_chk = gr.Checkbox(
                    value=USE_LLM_DEFAULT,
                    label="Use tiny LLM (agent mode, beta)"
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

        state = gr.State({"sheet": None})

        def on_user_send(history, text):
            history = history + [(text, None)]
            return history, ""

        def on_bot_reply(history, st, stage, use_llm):
            st, reply = run_pipeline(st, history[-1][0], stage, use_llm=bool(use_llm))
            history[-1] = (history[-1][0], reply)
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
