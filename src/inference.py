from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "my_model.keras")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model(MODEL_PATH)
class_names = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in-scale", "scratches"]

# FastAPI app
app = FastAPI()

# Templates and static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Save uploaded image
    file_location = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Prepare image
    img = image.load_img(file_location, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)/255.0

    # Predict
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = round(float(np.max(predictions)*100),2)
    prediction = class_names[class_index]

    # Return template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction,
        "confidence": confidence,
        "image_path": f"/static/uploads/{file.filename}"
    })

