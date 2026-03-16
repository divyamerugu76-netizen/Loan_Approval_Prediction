from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    salary = float(request.form["salary"])
    credit_score = float(request.form["credit_score"])
    loan_amount = float(request.form["loan_amount"])
    employment_status = int(request.form["employment_status"])

    prediction = model.predict([[salary, credit_score, loan_amount, employment_status]])
    result = "Approved" if prediction[0] == 1 else "Rejected"

    return render_template("index.html", prediction_text="Loan Status: " + result)

if __name__ == "__main__":
    app.run(debug=True)