---
title: Sepsis Spotter UI
emoji: 🚑
colorFrom: green
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

**Sepsis Spotter Project Repository Quick Links**

[Build](https://github.com/ffr0517/sepsis-spotter-build)
[API](https://github.com/ffr0517/sepsis-spotter-api)
[UI](https://github.com/ffr0517/sepsis-spotter-ui)
[Manuscript](https://github.com/ffr0517/sepsis-spotter-manuscript)

# Spot Sepsis – UI Repository

**Mobile-friendly research interface for interacting with the Spot Sepsis models (S1/S2).**  
**For research use only — not medical advice.**

---

## Current Implementation (Legacy-Compatible)

This version of the app connects to the **previous Spot Sepsis API build** (legacy endpoints).  
It provides a chat-based interface that collects clinical and laboratory information and runs the Spot Sepsis S1 and S2 models through either:

- **Hugging Face Spaces** (remote inference)  
- **Localhost (127.0.0.1)** for self-hosted testing

### Core Features
- Conversational intake assistant that builds a structured “Info Sheet.”  
- Username/password research access gate.  
- Sheet merge + resume for continuing sessions.  
- “Run S1” / “Run S2” buttons gated by data completeness.  
- Deterministic parser mode for use without an API key.

---

## Planned Upgrades (Next Major Integration Phase)

This UI will be refactored once the new `spot-sepsis-build`, and `spot-sepsis-api` repositories reach their stable release.

| Area | Planned Enhancement | Target Behaviour |
|------|--------------------|------------------|
| Input Handling | Dropdown, CSV upload, enhanced LLM-assisted entry | Multiple structured input paths |
| Validation | Automatic parity testing with backend outputs | ≥ 99–100 % equivalence before deployment |
| Testing | Red-team prompt and robustness testing | Identify edge/failure cases |
| Adaptive Learning | Opt-in anonymised “live-model” mode | Allows continual learning with clinic-controlled sharing |
| Version Governance | Benchmark-driven model promotion (PR-AUC criteria) | Only validated updates become active |
| User Analytics | Clinic dashboards and optional outcome logging | Local record-keeping + contribution toggle |
| Privacy Controls | “Share data with developers” toggle | Clear separation between private / shared data |

---

## Deployment

Automatic deployment is handled by `.github/workflows/deploy-to-space.yml`.

**Workflow summary**
1. Checks out this repository.  
2. Clones the target Hugging Face Space using `HF_SPACE` token + `SPACE_ID`.  
3. Syncs `app.py`, `requirements.txt`, `README.md` 
4. Commits and pushes any changes.