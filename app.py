# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback
import os

# Initialize Flask
app = Flask(__name__)
CORS(app)  # Allow requests from frontend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "crop_doctor_enhanced.pkl")
)

# Supported crops
SUPPORTED_CROPS = [
    "Rice", "Wheat", "Maize", "Sugarcane", "Cotton", "Soybean",
    "Groundnut", "Mango", "Banana", "Coconut", "Tomato", "Potato", "Chickpea"
]

# Mock icon mapping for severity
ICON_MAPPING = {
    "Low": "fa-solid fa-circle-check",
    "Medium": "fa-solid fa-triangle-exclamation",
    "High": "fa-solid fa-triangle-exclamation"
}

# Endpoint to fetch supported crops
@app.route("/crops", methods=["GET"])
def get_crops():
    return jsonify({"crops": SUPPORTED_CROPS})

@app.route("/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json()
    try:
        crop = data.get("crop")
        symptoms = data.get("symptoms")
        city = data.get("location", "")

        if not crop or not symptoms:
            return jsonify({"error": "Missing crop or symptoms"}), 400

        # Transform symptoms
        X_input = model.named_steps['tfidf'].transform([f"{crop} {symptoms} {city}"])
        distances, indices = model.named_steps['nn'].kneighbors(X_input)

        best_match_idx = indices[0][0]
        # Load disease info from your CSV database
        import pandas as pd
        db = pd.read_csv(os.path.join(BASE_DIR, "crop_diseases.csv"))
        best_match = db.iloc[best_match_idx]

        similarity_score = 1 - distances[0][0]
        severity = "Low" if similarity_score > 0.5 else "Medium" if similarity_score > 0.3 else "High"

        TREATMENT_DB = {
            "Early Blight": best_match['Organic_Remedy'],
            "Late Blight": best_match['Organic_Remedy'],
            "Leaf Spot": best_match['Organic_Remedy']
        }

        treatment = TREATMENT_DB.get(best_match['Disease'], "Consult local agronomist")

        return jsonify({
            "city": city,
            "crop": crop,
            "diagnosis": best_match['Disease'],
            "severity": severity,
            "confidence": round(similarity_score, 3),
            "treatment": treatment,
            "icon": ICON_MAPPING.get(severity, "fa-solid fa-circle-check")
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True, port=8000)