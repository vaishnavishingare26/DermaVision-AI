# DermaVision AI

Explainable AI-based skin lesion analysis project.

## Structure

- `frontend/` — React + Vite application
- `backend/` — Flask API
- `ai_model/` — dataset, preprocessing, classification, segmentation, metadata fusion and Grad-CAM
- `reports/` — generated reports/templates
- `database/` — database schema

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

## Backend

```powershell
cd backend
python app.py
```

Do not place HAM10000 or ISIC 2024 datasets inside the frontend. Keep them under `ai_model/dataset/`.
