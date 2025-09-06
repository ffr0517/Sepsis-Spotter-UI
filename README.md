# Sepsis-Spotter-UI (Gradio)

Mobile-friendly, password-gated UI that helps a user provide inputs and calls the Spot Sepsis API (S1/S2). For research only — not medical advice.

## Local dev
```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export SEPSIS_API_URL_S1="https://YOUR-RENDER-API/s1_infer"
export SEPSIS_API_URL_S2="https://YOUR-RENDER-API/s2_infer"
python app.py

🚀 Test deploy at 2025-09-06