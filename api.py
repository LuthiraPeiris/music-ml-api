from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)
model = joblib.load("music_recommender.joblib")

@app.get("/")
def home():
    return "Music ML API running"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(force=True)

    age = int(data["age"])
    gender = int(data["gender"])

    print(f"Request received → age={age}, gender={gender}")

    genre = model.predict([[age, gender]])[0]

    print(f"Prediction → {genre}")

    return jsonify({"genre": genre})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)