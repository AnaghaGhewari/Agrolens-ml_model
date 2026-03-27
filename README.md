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
