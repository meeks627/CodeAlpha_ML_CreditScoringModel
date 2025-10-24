import numpy as np
import pandas as pd
import pickle

def load_preprocessor(path='models/preprocessor.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model(path='models/tuned_xgb_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model_params(path='models/tuned_xgb_model_parameters'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def make_prediction(sample, preprocessor, model):
    sample_preprocessed = preprocessor.transform(sample)
    prediction = model.predict(sample_preprocessed)
    # Get prediction probabilities
    if hasattr(model, "predict_proba"):
        prediction_probs = model.predict_proba(sample_preprocessed)[0]
        print('Probalities of getting a loan [Not worthy   Worthy]: ', prediction_probs)
        # Determine the probability of the predicted class
        predicted_proba = prediction_probs[prediction].item() * 100 # Convert to percentage

        if prediction == 1:
            predicted_proba = predicted_proba
#             print(f"""You are eligible for a loan. 
# You are {predicted_proba:.2f}% worthy of this loan.""")
        else:
          predicted_proba = 100 - predicted_proba
#           print(f"""
# You are not eligible for a loan. 
# Your creditworthiness is {predicted_proba:.2f}% which is less than average of 50%.
# We're very sorry we couldn't help you.""")

    else:
        print("Model does not support predict_proba method.")
    return  prediction.item(),predicted_proba



# The manual testing of the model

def result_summary(sample):
    if __name__ == "__main__":
        preprocessor = load_preprocessor()
        model = load_model()
        # Example sample for prediction
    
        sample_df = pd.DataFrame(sample, columns=['person_age', 'person_income', 'loan_amnt', 'person_emp_length',
                                                'loan_int_rate','loan_percent_income','cb_person_cred_hist_length',
                                                'person_home_ownership','loan_intent','loan_grade',
                                                'cb_person_default_on_file'])
        prediction,predicted_proba = make_prediction(sample_df, preprocessor, model)
        return f'Prediction, probability : {prediction},{predicted_proba:.2f}%'


sample_for_not_worthy = np.array([[21, 9600, 1000, 5,11.14,0.1,1,'OWN','EDUCATION','B',2]])
sample_for_worthy = np.array([[35, 1000, 15000, 5, 12.5, 20, 10, 'RENT', 'PERSONAL', 'B', 0]])
print(result_summary(sample_for_worthy))