# DermaVision AI Frontend — Corrected

## Features
- Live connection to Flask `POST /api/predict`
- No hardcoded Melanoma/94.72% prediction
- English, Hindi and Marathi UI language support
- New Patient / Existing Patient workflow
- Automatic unique Patient ID such as `DV-P-00001`
- Existing Patient ID lookup
- Patient profile + prediction history persisted in browser localStorage
- Actual backend confidence, risk and class probabilities shown on the result page
- Printable report with Patient ID

## Run
1. Start Flask from the project root:
   `python -m backend.app`
2. Open a second terminal and enter the frontend folder:
   `cd frontend`
3. Install dependencies:
   `npm install`
4. Start Vite:
   `npm run dev`
5. Open the URL shown by Vite, normally `http://localhost:5173`

The frontend defaults to:
`http://127.0.0.1:5000/api`

To change it, create `.env` from `.env.example` and set:
`VITE_API_BASE_URL=http://127.0.0.1:5000/api`

## Important
The current patient storage is browser-local for this frontend-only integration. For a multi-user production deployment, connect the patient records to a server-side database/API so records are shared across users/devices.

The AI result is research decision support and is not a medical diagnosis.
