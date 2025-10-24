import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
np.random.seed(42)

class PreProcess():
  def __init__(self,numeric_features = ['person_age', 'person_income', 'loan_amnt', 'person_emp_length',
                          'loan_int_rate','loan_percent_income','cb_person_cred_hist_length'],
                    categorical_features = ['person_home_ownership','loan_intent'],
                    ordinal_features = ['loan_grade']):
    
    self.numeric_features = numeric_features
    self.categorical_features = categorical_features
    self.ordinal_features = ordinal_features
    

  def preprocess_credit_data(self,df):
      df = df.copy()
      # Fill missing values
      df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
      df['loan_int_rate'] = df['loan_int_rate'].apply(lambda x: np.random.uniform(5,20) if np.isnan(x) else x)
      # Binary encode
      df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y':1,'N':0})

      X = df.drop('loan_status', axis=1)
      y = df['loan_status']
      grade_order = [['A','B','C','D','E','F','G']]

      preprocessor = ColumnTransformer(
          transformers=[
              ('num', StandardScaler(), self.numeric_features),
              ('cat', OneHotEncoder(drop='first'), self.categorical_features),
              ('ord', OrdinalEncoder(categories=grade_order), self.ordinal_features)
          ],
          remainder='passthrough'
      )
      return X, y, preprocessor


      # Getting the preprocessed dataframe
  def preprocess_dataset(self,df):
    x,y,preprocessor = self.preprocess_credit_data(df)
    X_preprocessed = preprocessor.fit_transform(x)
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
    all_columns = (self.numeric_features +
                    list(encoded_cat_names) + 
                    self.ordinal_features +
                    ['cb_person_default_on_file'])
    X_preprocessed_df = pd.DataFrame(X_preprocessed,columns=all_columns)
    return X_preprocessed_df,x,y,preprocessor