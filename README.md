FastAPI MNIST Digit Classifier

This project demonstrates how to deploy a trained Machine Learning model (a Random Forest Classifier for handwritten MNIST digits) using a high-performance Python web framework, FastAPI. The application includes a simple HTML frontend for users to upload images and receive predictions.

The core functionality of this project is:

Training a model to recognize handwritten digits (0-9).

Serving that model via a REST API endpoint (/predict/).

Providing a responsive web interface (index.html) to interact with the API.

🚀 Getting Started

Follow these steps to set up and run the project locally.

Prerequisites

You need Python 3.8+ installed on your system.

1. Project Setup and Environment

First, create and activate a virtual environment to manage dependencies.

# 1. Navigate to your project directory
cd fastapi-mnist-app

# 2. Create the virtual environment
python -m venv venv

# 3. Activate the environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
# source venv/bin/activate


2. Install Dependencies

Install all necessary packages using the requirements.txt file (or the command below):

# If you created requirements.txt:
# pip install -r requirements.txt

# Or install directly:
pip install fastapi "uvicorn[standard]" scikit-learn numpy pillow python-multipart


3. Train the Model

The main application (main.py) requires a pre-trained model file named mnist_model.pkl. You must run the training script once.

python mlmodel.py


(This script will download the MNIST dataset, train a Random Forest classifier, and save the resulting model as mnist_model.pkl.)

Note on Model Size: The mnist_model.pkl file is intentionally excluded from Git tracking via .gitignore because it exceeds GitHub's file size limit (100MB). It must be generated locally.

4. Run the FastAPI Server

Once the model is trained, start the FastAPI application using Uvicorn.

python -m uvicorn main:app --reload


🌐 Usage

Open your web browser and navigate to the local address displayed in the terminal (e.g., http://127.0.0.1:8000).

The index.html page will load.

Upload a square, grayscale image containing a handwritten digit (for best results, try to mimic the MNIST style: white digit on a black background, or simply use a clear, centered digit).

Click Predict Digit to receive the model's prediction.

API Endpoint

The raw prediction endpoint can be tested independently:

Method

URL

Description

POST

/predict/

Accepts a multipart form data file (UploadFile) and returns a JSON object with the predicted digit.

Example Response:

{
  "prediction": "7"
}


📦 Project Files Overview

File

Description

main.py

The core FastAPI application. Handles serving the HTML, loading the model, and defining the /predict/ endpoint.

train_model.py

Script used to download the MNIST data, train the Random Forest model, and save it as mnist_model.pkl.

index.html

The HTML frontend with JavaScript for file handling and interacting with the FastAPI endpoint.

requirements.txt

Lists all necessary Python dependencies.

.gitignore

Ensures development files (venv/, mnist_model.pkl) are not committed to Git.
