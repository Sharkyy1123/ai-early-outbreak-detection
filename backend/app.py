from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/anomaly_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Make sure we extract a SINGLE number
    cases = float(data["cases"])

    # Model expects 2D array
    prediction = model.decision_function([[cases]])

    risk_score = float((1 - prediction[0]) * 100)

    return jsonify({"risk_score": risk_score})

if __name__ == "__main__":
    app.run(debug=True)
