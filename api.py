from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app) 

# Load model once at startup
model = joblib.load("music_recommender.joblib")

@app.get("/")
def health():
    return "Music ML API running"

@app.post("/predict")
def predict():
    data = request.get_json(force=True)

    age = int(data["age"])
    gender = int(data["gender"])

    genre = model.predict([[age, gender]])[0]
    return jsonify({"genre": genre})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)