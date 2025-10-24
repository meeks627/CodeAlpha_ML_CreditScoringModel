import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,f1_score

def train_models(model,x_train, y_train, x_test, y_test,verbose=True):
    # fit model
    model.fit(x_train,y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    results = {
        'model': model,
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_pred)
    }
    results = pd.Series(results,index=results.keys())
    return results
