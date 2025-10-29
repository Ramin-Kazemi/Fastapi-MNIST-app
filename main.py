import io
import numpy as np
import pickle
from typing import List

# Import PIL for image processing
import PIL.Image
import PIL.ImageOps

# Import FastAPI components
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Model Loading ---
try:
    with open('mnist_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("MNIST Model loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'mnist_model.pkl' not found. Please run the ML script first.")
    # Exit or handle the error appropriately
    model = None

# --- FastAPI Initialization and Configuration ---
app = FastAPI(title="MNIST Digit Classifier")

# To serve static files (like your index.html)
# We will use Jinja2Templates for simplicity instead of StaticFiles for index.html
# Create a 'templates' directory and place index.html inside it
# For this example, we'll keep it simple and serve it directly via HTMLResponse

# If you wanted a more complex setup:
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# You only need CORSMiddleware if your frontend and backend were on different domains/ports.
# Since we are serving the HTML from FastAPI itself, it's often not strictly necessary,
# but it's good practice for general APIs.
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- Utility Function for Image Preprocessing ---
def preprocess_image(image_data: bytes) -> List[float]:
    """Converts the uploaded image into the 784-feature vector the model expects."""
    image = PIL.Image.open(io.BytesIO(image_data))

    # Convert to grayscale
    image = image.convert('L')
    
    # Invert colors (MNIST is white digit on black background)
    image = PIL.ImageOps.invert(image)

    # Resize to 28x28 (MNIST standard size)
    image = image.resize((28, 28))

    # Convert to numpy array and normalize
    # The image is 28x28, total 784 pixels
    img_array = np.array(image).flatten().astype('float32') / 255.0
    
    # The model expects a list/array of 784 features
    return img_array.tolist()

# --- Routes ---

# 1. Root route to serve the HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serves the index.html file."""
    try:
        with open("index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found!</h1>", status_code=404)

# 2. Prediction endpoint
@app.post("/predict/")
async def predict_digit(file: UploadFile = File(...)):
    """Accepts an image file and returns the predicted digit."""
    if model is None:
        return {"error": "Model not loaded. Check server logs."}
        
    # Read the file content
    image_bytes = await file.read()
    
    # Preprocess the image
    try:
        processed_features = preprocess_image(image_bytes)
    except Exception as e:
        return {"error": f"Image processing failed: {e}"}

    # Make the prediction
    # The model expects a 2D array: [[feature_1, feature_2, ..., feature_784]]
    features_2d = np.array(processed_features).reshape(1, -1)
    
    prediction = model.predict(features_2d)[0]
    
    return {"prediction": str(prediction)}