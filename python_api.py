
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the saved emotion model
model = joblib.load("model/text_emotion.pkl")

# Map model emotions to MoodTag's five moods
MOOD_MAP = {
    "joy": "Happy",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "sadness": "Sad",
    "anger": "Angry",
    "fear": "Anxious",
    "disgust": "Angry",
    "shame": "Anxious",
    "surprise": "Happy",
}

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "MoodTag Python API is running!"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    prediction = str(model.predict([text])[0]).lower()
    mood = MOOD_MAP.get(prediction, "Neutral")

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = round(float(max(probabilities)) * 100, 2)

    return jsonify({
        "mood": mood,
        "originalEmotion": prediction,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

