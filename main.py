from src.preprocessing import PreProcess
from src.utils import set_global_seed
from src.train_models import train_models
from src.config import SEED, XGB_PARAM_GRID
from src.eda import check_dataset,plot_correlation, plot_feature_distributions
from src.hyperparameter_tunning import hypertune
from src.metrics_visualization import compare_models, PlotMetrics

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle





def main():
	# Make runs deterministic
	set_global_seed(SEED)

	# load and preprocess data
	df = pd.read_csv('data/credit_risk_dataset.csv')
	print('Dataset before Preprocessing:')
	check_dataset(df)

	preprocess = PreProcess() # instantiating the class for preprocessing
	Processed_dataset,X,y,processor = preprocess.preprocess_dataset(df=df)
	print(Processed_dataset.head())

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
	X_train_preprocessed = processor.fit_transform(X_train)
	X_test_preprocessed = processor.transform(X_test)

	# EDA after Preprocessing
	print('\n\tDataset after Preprocessing:')
	check_dataset(Processed_dataset)
	plot_feature_distributions(Processed_dataset)
	plot_correlation(Processed_dataset)


	# Train models
	logreg = LogisticRegression(max_iter=1000, random_state=SEED)
	svc = SVC()
	xgb = XGBClassifier(random_state=SEED, seed=SEED)
	logreg_results = train_models(logreg,X_train_preprocessed, y_train, X_test_preprocessed, y_test)
	svc_results = train_models(svc,X_train_preprocessed, y_train, X_test_preprocessed, y_test)
	xgb_results = train_models(xgb,X_train_preprocessed, y_train, X_test_preprocessed, y_test)
	result = pd.DataFrame([logreg_results, xgb_results,svc_results], index=['Logistic Regression', 'XGBoost','svmClassifier'])

	# comparing models with test data
	print("\nComparison of Model Results:\n", result)
	models = {
		'Logistic Reg': logreg,
		'SVM': svc,
		'XGBoost': xgb
	}
	compare_models(models, X_test_preprocessed, y_test)


	# Hyperparameter tuning
	bestmodel,bestparam,bestscore = hypertune(
		model=XGBClassifier(),
		param_grid=XGB_PARAM_GRID,
		x_train=X_train_preprocessed,
		y_train=y_train
		)

	# Retrain highest performing hypertuned model on full training data
	tuned_xgb = bestmodel
	tuned_xgb_results = train_models(tuned_xgb,X_train_preprocessed, y_train, X_test_preprocessed, y_test)
	print('Tuned XGBoostClassifier Results: \n', tuned_xgb_results)

	# Plot metrics for the tuned model
	pm = PlotMetrics(tuned_xgb,X_test_preprocessed,y_test)
	pm.plot_confusion_matrix()
	pm.roc_auc_curve()

	# Saving the trained model
	with open('models/preprocessor.pkl', 'wb') as f:
					pickle.dump(processor, f)
	with open('models/tuned_xgb_model.pkl', 'wb') as f:
					pickle.dump(tuned_xgb, f)
	with open('models/tuned_xgb_model_parameters', 'wb') as f:
					pickle.dump(bestparam, f)




if __name__ == "__main__":
        main()