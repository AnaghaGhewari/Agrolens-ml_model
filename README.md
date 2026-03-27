# Smart Farming ML Model 🌱

## Overview

This repository contains a PyTorch-based machine learning model developed for smart farming applications. The model analyzes agricultural data to provide accurate predictions that help in making cost-effective and eco-friendly farming decisions.

## Features

* Data preprocessing and model training
* Prediction system using trained model
* Easy integration with backend APIs

## Tech Stack

* Python
* PyTorch
* NumPy, Pandas

## Files

* train.ipynb – Model training notebook
* app.py – Model inference code
* requirements.txt – Dependencies

## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run the application:
   python app.py

## Note

This repository is a part of a larger smart farming system integrating ML, backend, and frontend components.



# 🌿 Plant Disease Detection API

## 🚀 How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Start server:
   uvicorn app:app --reload

3. Open in browser:
   http://127.0.0.1:8000/docs

---

## 📡 API Endpoint

POST /predict

Input:

* Upload image file (key = "file")

Output:
{
"disease": "Tomato - Early blight",
"confidence": 0.91
}

---

## 🧠 Model Info

* Model: EfficientNet B0 (PyTorch)
* Classes: 38 plant diseases
* Input: RGB image
* Output: Disease + confidence score
