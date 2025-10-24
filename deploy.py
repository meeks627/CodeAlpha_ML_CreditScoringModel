import requests
url = 'http://127.0.0.1:5000/predict'

data = {
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

headers = {'Content-Type': 'application/json'}
response = requests.post(url,json=data,headers=headers)

print(response.json())