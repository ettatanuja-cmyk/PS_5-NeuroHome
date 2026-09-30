# BCI Home Appliance Prediction System (EEG-TCFNet)

Full-stack app: **FastAPI backend + React/Vite/Tailwind dashboard** for the
Home-Appliance-Control ERP-BCI dataset.

## Layout

```
backend/
  main.py                  # FastAPI service (POST /api/predict)
  requirements.txt
  services/data_loader.py  # loadmat + zip ingestion + 120Hz / 1-15Hz / normalize
  models/eeg_tcfnet.py     # EEGNet -> TCN -> 2xLSTM(30) -> Fuzzy -> Softmax
frontend/                  # Vite + React + TS + Tailwind + Recharts + lucide-react
Home-Appliance-Control-Dataset-main/   # dataset (sig_vec/trigger, cal_sig, param.mat)
```

## Dataset download (not in git — 3.9 GB)

This repo excludes `Home-Appliance-Control-Dataset-main/` via `.gitignore`
(GitHub 100 MB/file limit). Download it separately:

- Source: https://github.com/jml226/Home-Appliance-Control-Dataset
- Paper: Lee et al. 2024, Front. Hum. Neurosci. 18:1320457
- Place it at project root as `Home-Appliance-Control-Dataset-main/` so the
  backend loader finds `sig_vec/trigger`, `cal_sig`, `param.mat`.

```bash
# project root
git clone https://github.com/jml226/Home-Appliance-Control-Dataset.git Home-Appliance-Control-Dataset-main
```

## Dataset notes (verified)

- Trial files: `sig_vec` (channels×time, e.g. 25×30240) + `trigger` (1×time).
  Trigger codes: 11 block start, 12 stim start, 13 block end, 1–6 stimulus types.
- Calibration files: `cal_sig` (channels×time, no trigger).
- Recording rate 500 Hz (`param.mat`: `Fs=500`, `dFs=125`); channel count varies
  (e.g. AC 25 ch, TV 31 ch) — the loader/model handle any layout.

## Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000   # run from inside backend/
```

Endpoints:

| Method | Path | Description |
|---|---|---|
| POST | `/api/predict` | multipart `file` (.mat or .zip); query `aggregate=first\|mean`, `orig_fs=500`, `appliance=` (optional label context, e.g. `TV`) |
| GET | `/api/health`, `/api/classes` | status / class list |
| GET | `/api/appliances` | per-appliance catalog: commands, paradigm, stimuli, subjects |
| POST | `/api/predict_path?path=...` | dev helper for server-local files |

Response shape:

```json
{
  "predicted_appliance": "Electric Light",
  "class_id": 4,
  "confidence_scores": {"Air Conditioner": 0.03, "Bluetooth Speaker": 0.05, "Door Lock": 0.02, "Electric Light": 0.88, "TV": 0.02},
  "p300_detected": true,
  "predicted_command": {
    "stimulus_id": 3, "label": "Channel 3", "confidence": 0.89,
    "command_scores": {"Channel 1": 0.02, "Channel 2": 0.04, "Channel 3": 0.89, "Channel 4": 0.05},
    "context_appliance": "TV"
  },
  "signal_preview": [/* 300 time points, Ch1 */]
}
```

Set `EEG_TCFNET_CKPT=/path/to/model.pth` to use trained torch weights instead.
Install `torch` to enable the full `EEGTCFNet` graph.

### Appliance classifier (trained, default)

`backend/train_appliance_clf.py` trains a multinomial logistic regression on
16 per-file features (channels, duration, trigger/stimulus counts, amplitude
stats, spectral shape) and saves `backend/models/appliance_clf.npz`, which the
API loads automatically (`model_source: trained-ml ...` in responses).

Retrain any time from inside `backend/`: `python train_appliance_clf.py`

Measured accuracy: ~85-89% on held-out subjects (Air Conditioner /
Bluetooth Speaker / TV near-perfect; occasional Door Lock ↔ Electric Light
confusion — both are 4-class tablet paradigms whose EEG is near-identical).

## Frontend

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (proxies /api -> :8000)
npm run build    # production bundle in dist/
```

Dashboard: drag-drop upload → predicted appliance + P300 badge → confidence
bars → multi-channel preprocessed EEG line chart → archive file list.
