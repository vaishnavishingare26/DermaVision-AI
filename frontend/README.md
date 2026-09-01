# DermaVision AI Frontend - Improved

## Run
```bash
npm install
npm run dev
```

## What is included
- Premium medical AI login
- Dashboard
- Patient metadata form
- Actual image upload + preview + validation
- Prediction result screen
- Segmentation/Grad-CAM placeholders ready for backend outputs
- Patient history
- Analytics
- Clinical-style report + print/PDF
- Responsive UI
- Flask API integration point documented for later backend connection

## Backend contract planned
`POST http://localhost:5000/api/predict` with multipart form data:
`image` + patient metadata.

The current prediction is intentionally demo data until the trained model is connected; the UI does not pretend that the demo score is a real model result.
