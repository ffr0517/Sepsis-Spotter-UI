import os, re, json, time, requests, gradio as gr

# ---- Optional tiny LLM for parsing (CPU) ----
USE_LLM = False
_llm_pipe = None

def _load_tiny_llm():
    global _llm_pipe
    if _llm_pipe is not None:
        return _llm_pipe
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id)
        _llm_pipe = pipeline(
            "text-generation",
            model=mdl,
            tokenizer=tok,
            # keep it small for CPU
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
        )
        return _llm_pipe
    except Exception as e:
        print("LLM load failed, falling back to regex:", e)
        _llm_pipe = None
        return None

SYSTEM_PROMPT = (
    "You are a clinical intake assistant. Extract only structured fields needed for a sepsis risk model.\n"
    "Return STRICT JSON with keys:\n"
    "{\n"
    '  "clinical": {\n'
    '    "age.months": <number>, "sex": <0 for male, 1 for female>, "hr.all": <int>,\n'
    '    "rr.all": <int>, "oxy.ra": <int>, ... (optional extras)\n'
    "  },\n"
    '  "labs": { "CRP": <number>, "PCT": <number>, "Lactate": <number>, "WBC": <number>, "Neutrophils": <number>, "Platelets": <number> }\n'
    "}\n"
    "Rules:\n"
    "- Convert years to months.\n"
    "- For sex, map male/boy→0, female/girl→1.\n"
    "- If a value is missing, omit the key (do not invent values).\n"
    "- Output ONLY JSON. No commentary.\n"
)

def llm_extract_to_dict(user_text: str) -> dict:
    if not USE_LLM:
        return {}
    pipe = _load_tiny_llm()
    if pipe is None:
        return {}
    prompt = (
        SYSTEM_PROMPT
        + "\n\nUser text:\n"
        + user_text.strip()
        + "\n\nJSON:"
    )
    out = pipe(prompt)[0]["generated_text"]
    # Try to locate a JSON object in the output
    import re, json
    m = re.search(r"\{[\s\S]*\}\s*$", out)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
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
# Simple rule-based extractor (robust + free)
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
# Orchestration
# --------------------------------
def run_pipeline(state, user_text, stage="auto"):
    # 1) Try LLM extraction
    blob = llm_extract_to_dict(user_text or "")
    clin_new = (blob.get("clinical") or {}) if isinstance(blob, dict) else {}

    # 2) Fallback to regex for anything missing
    rx_clin, rx_labs, _ = extract_features(user_text or "")
    # Merge (LLM has priority; regex fills gaps)
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

    # Validate S1 inputs if running S1
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
# Gradio UI
# --------------------------------
with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("### Spot Sepsis — Research Preview *(Not medical advice)*")
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=420, type="tuples")
            msg = gr.Textbox(placeholder="Describe the case (e.g., '2-year-old, HR 154, RR 36, SpO₂ 95%')", lines=3)
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

    def on_bot_reply(history, st, stage):
        st, reply = run_pipeline(st, history[-1][0], stage)
        history[-1] = (history[-1][0], reply)
        info_json = json.dumps(st["sheet"], indent=2) if st.get("sheet") else ""
        return history, st, info_json, ""

    def on_merge(st, pasted):
        try:
            blob = json.loads(pasted)
        except Exception:
            return st, "Could not parse pasted JSON.", ""
        if st.get("sheet"):
            st["sheet"] = merge_sheet(st["sheet"], blob.get("features", {}).get("clinical", {}), blob.get("features", {}).get("labs", {}))
        else:
            st["sheet"] = blob
        return st, "Merged.", json.dumps(st["sheet"], indent=2)

    msg.submit(on_user_send, [chat, msg], [chat, msg]).then(
        on_bot_reply, [chat, state, gr.State("auto")], [chat, state, info, msg]
    )
    btn_s1.click(on_user_send, [chat, msg], [chat, msg]).then(
        on_bot_reply, [chat, state, gr.State("S1")], [chat, state, info, msg]
    )
    btn_s2.click(on_user_send, [chat, msg], [chat, msg]).then(
        on_bot_reply, [chat, state, gr.State("S2")], [chat, state, info, msg]
    )
    btn_auto.click(on_user_send, [chat, msg], [chat, msg]).then(
        on_bot_reply, [chat, state, gr.State("auto")], [chat, state, info, msg]
    )
    merge_btn.click(on_merge, [state, paste], [state, tips, info])

# ---- Launch settings: Spaces vs local ---------------------------------
# ---- Auth via function (avoids proxy/basic-auth loops on Spaces) ----
SPACE_USER = os.getenv("SPACE_USER", "user")   # set in Space → Settings → Variables & secrets
SPACE_PASS = os.getenv("SPACE_PASS", "pass")

def auth_fn(username, password):
    return (username == SPACE_USER) and (password == SPACE_PASS)

ON_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("SYSTEM") == "spaces")

if ON_SPACES:
    # Let Spaces handle host/port; disable SSR to avoid i18n issue
    demo.launch(auth=auth_fn, ssr_mode=False)
else:
    demo.launch(server_name="127.0.0.1", server_port=7860, auth=auth_fn)