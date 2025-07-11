# FCGTN
Minimal reproducible example (MRE) for the “Federated Causal-Graph Transfer Network” paper.   Includes synthetic demo data and a lightweight inference script; real industrial data remain confidential.


# FCGTN • Minimal Reproducible Example (MRE)

This repository provides a **small, runnable demo** of the methods described in  Paper

**Disclaimer
* The CSV contained here is **synthetic**. Values are randomly drawn so that mean/variance roughly match the real sensor streams, **but no commercial or personal information is included**.  
* We publish this MRE solely to satisfy the **Data Availability Policy** and to help reviewers reproduce the main claims.  
* Full production data and full-size model weights remain confidential under NDA with our industrial partners.

---

## Repository contents

| File | Purpose |
|------|---------|
| `synthetic_data.csv` | 10 000 rows × 5 channels of fake sensor readings |
| `model.pt` | Tiny single-layer LSTM checkpoint (~150 kB) for demo only |
| `inference_demo.py` | Script that loads the model, reads the CSV and outputs a 30-min forecast (μ, σ) |
| `requirements.txt` | Open-source dependencies (`torch`, `numpy`, `pandas`) |

---

## Quick start

```bash
# 1. set up a clean environment
python -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the demo
python inference_demo.py \
       --csv synthetic_data.csv \
       --weights model.pt \
       --out forecast.json
