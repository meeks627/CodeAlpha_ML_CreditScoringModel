import pandas as pd
from manual_tests.making_predictions import load_model,load_preprocessor,make_prediction
from flask import Flask, request, jsonify

app = Flask(__name__)
creditmodel = load_model()
creditpreprocessor = load_preprocessor()

@app.route("/")
def home():
    return "Creditworthiness App is working"

@app.route("/predict",methods=['POST'])
def predict():
    try:
        # get json data from api request
        data = request.get_json()
        input_data = pd.DataFrame([data])

        # check if input data is provided
        if not data:
            return jsonify({
                "error": "Input data not provided"
            }),400

        required_columns =[
                            'person_age', 'person_income', 'loan_amnt', 'person_emp_length',
                            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                            'person_home_ownership', 'loan_intent', 'loan_grade',
                            'cb_person_default_on_file'
]
        # check if all columns are provided
        if not all (col in input_data.columns for col in required_columns):
            return jsonify({
                "error": f"Not all columns has been provided, Kindly check again. Here are the required columns {required_columns}"
            }),400
    
        prediction, probability  = make_prediction(input_data,creditpreprocessor,creditmodel)

        # response for the api
        if prediction == 1:
            message = f"You are eligible for a loan. You are {probability:.2f}% worthy of this loan."
        else:
          probability = 100 - probability
          message = f"You are not eligible for a loan. Your creditworthiness is {probability:.2f}% which is less than average of 50%. We're very sorry we couldn't help you."
        
        response = {
            'prediction': int(prediction),
            'Probablility': round(probability,2),
            'message': message
        }
        # return the jsonify
        return jsonify(response)
    except Exception as e:
        return jsonify({"error":str(e)}),500
    

if __name__ == "__main__":
    app.run(debug=True)