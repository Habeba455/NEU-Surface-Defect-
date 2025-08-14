## 🏭 NEU Surface Defect Detection

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

⚡ Quick Start

pip install -r requirements.txt

<img width="1431" height="653" alt="Screenshot 2025-08-15 at 2 13 46 AM" src="https://github.com/user-attachments/assets/eeeee062-edb1-4787-9f3e-a73e836c95c6" />Run FastAPI server

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





