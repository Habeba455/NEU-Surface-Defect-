<img width="550" height="358" alt="image" src="https://github.com/user-attachments/assets/6b501744-2a0c-48a4-aa15-135e33b86327" /># 🏭 NEU Surface Defect Detection
Detecting and classifying industrial **surface defects** using a **Convolutional Neural Network (CNN)**.  
This project provides both a **FastAPI backend** and a **web interface** for uploading images and getting predictions.

---
<img width="550" height="358" alt="image" src="https://github.com/user-attachments/assets/fd5c9994-9f77-4dac-9bf4-73eeaf25ab27" />
## 🚀 Features
- Trained CNN model on the **NEU-DET dataset**  
- Classifies defects into 6 categories:
  - Crazing  
  - Inclusion  
  - Patches  
  - Pitted Surface  
  - Rolled-in Scale  
  - Scratches  
- Web interface for uploading images  
- REST API with **FastAPI**  
- Deployment ready with **Docker / Kubernetes**  

---

## 📂 Project Structure
```bash
├── data/                 # Dataset (not included in repo)
├── models/               # Saved trained models
├── src/
│   ├── api/              # FastAPI application
│   ├── templates/        # HTML files
│   ├── static/           # CSS files
│   ├── train.py          # Model training
│   ├── inference.py      # Run inference
│   └── evaluate.py       # Model evaluation
├── infra/                # Docker & Kubernetes configs
├── scripts/              # Deployment scripts
├── requirements.txt      # Dependencies
└── README.md             # Project description
⚡ Quick Start

pip install -r requirements.txt
![Uploading Screenshot 2025-08-15 at 2.13.46 AM.png…]()

Run FastAPI server
uvicorn src.api.app:app --reload


Web app: 👉 http://127.0.0.1:8000

API docs: 👉 http://127.0.0.1:8000/docs

📊 Model Performance

Accuracy: 95%

🛠️ Tech Stack

Python 3.9

TensorFlow / Keras

FastAPI

HTML / CSS

Docker, Kubernetes

GitHub Actions (CI/CD)
