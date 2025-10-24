#  Credit Worthiness Prediction

This project predicts whether a loan applicant is likely to receive loan or not and the credit worthiness (probability of receiving loan) of the applicant. It uses machine learning techniques and an XGBoost model fine tuned with GridSearchCV, and Flask API for serving predictions


---

```
Feature names and Descriptions 
person_age                 >>>>        Age
person_income              >>>>        Annual Income
person_home_ownership      >>>>        Home ownership
person_emp_length          >>>>        Employment length (in years)
loan_intent                >>>>        Loan intent
loan_grade                 >>>>        Loan grade
loan_amnt                  >>>>        Loan amount
loan_int_rate              >>>>        Interest rate
loan_status                >>>>        Loan status (0 is not eligible 1 is eligible) (Target class)
loan_percent_income        >>>>        Percent income
cb_person_default_on_file  >>>>        Historical default
cb_preson_cred_hist_length >>>>        Credit history length
```

##  Project Structure

---
## 🧠 Workflow
1. **Exploratory Data Analysis:**
    - Checked for missing values
    - Reviewed Statistics like mean, skewness, median etc.

2. **Preprocessing:**  
   - Handles missing values, encoding, and scaling.  
   - Saves preprocessor as `preprocessor.pkl`.

3. **Model Training & Tuning:**  
   - Uses Logistic Regression, SVM, and XGBoost.  
   - GridSearchCV for hyperparameter optimization.  
   - Best models saved in `/models`.

4. **Evaluation:**  
   - Accuracy, F1, Recall, and ROC-AUC.  
   - Visual comparisons using Seaborn plots.

5. **Prediction:**  
   -Test predictions via `tests/making_predictions.py`.
   - Flask API deployment handled by `deploy.py`

---

## 📈 Results of Each Model
```
    model           train_accuracy  test_accuracy  f1_score  precision    recall   roc_auc
Logistic Regression        0.854819       0.845174  0.583918   0.722449  0.489965  0.718169
XGBoost                    0.956722       0.933098  0.830350   0.948444  0.738408  0.863486
svmClassifier              0.916091       0.907626  0.755682   0.913641  0.644291  0.813470
```

**BEST MODEL:** XGBoost

---

## ⚙️ Setup & Run

```bash
# Clone repo
git clone https://github.com/meeks627/CodeAlpha_ML_CreditScoringModel.git
cd CodeAlpha_ML_CreditScoringModel

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run project
python main.py

# Run Flask API
python app.py
python deploy.py
```
```
# Example of Input data
 {
    'person_age':22,
    'person_income': 3200, 
    'loan_amnt':23, 
    'person_emp_length':12,                       
    'loan_int_rate':12.3, 
    'loan_percent_income': 5,
    'cb_person_cred_hist_length':1,
    'person_home_ownership': 'OWN',
    'loan_intent': 'PERSONAL',
    'loan_grade': 'B',
    'cb_person_default_on_file': 1
}

Output Response

{'Probablility': 99.88, 
'message': 'You are eligible for a loan. You are 99.88% worthy of this loan.', 
'prediction': 1
}

Next Steps
- Learn to deploy API to AWS
PS: Free for Frontend to work something around to make the predictions more user friendly


Author
Benedict Odiwe
odiwebenedict@gmail.com
https://www.linkedin.com/in/emeka-odiwe
```
