import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score,\
      recall_score, roc_auc_score, f1_score,confusion_matrix,\
      auc,roc_curve



def compare_models(models_dict, X_test, y_test, metrics=None):
    """
    Compare multiple models on the same test data using specified metrics.
"""
    if metrics is None:
        metrics = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']

    results = {metric: [] for metric in metrics}
    model_names = []

    for name, model in models_dict.items():
        model_names.append(name)
        y_pred = model.predict(X_test)
        y_probs = model.decision_function(X_test) if hasattr(model, "decision_function") else model.predict_proba(X_test)[:,1]
    

        # Calculate metrics
        for metric in metrics:
            if metric == 'accuracy':
                results[metric].append(accuracy_score(y_test, y_pred))
            elif metric == 'f1':
                results[metric].append(f1_score(y_test, y_pred))
            elif metric == 'precision':
                results[metric].append(precision_score(y_test, y_pred))
            elif metric == 'recall':
                results[metric].append(recall_score(y_test, y_pred))
            elif metric == 'roc_auc':
                results[metric].append(roc_auc_score(y_test, y_probs))

    # Plot each metric
    n_metrics = len(metrics)
    _, axes =  plt.subplots(1, n_metrics, figsize=(6*n_metrics,6))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes,metrics):
        sns.barplot(x=model_names, y=results[metric], ax=ax)
        ax.set_ylim(0,1)
        ax.set_title(f'Model Comparison: {metric}')
        ax.set_ylabel(metric)
        ax.set_xlabel('Models')
        
    plt.show()

    

class PlotMetrics():
    def __init__(self,model,x_test,y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def plot_confusion_matrix(self):
        y_pred = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test,y_pred)
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def roc_auc_curve(self):
        y_probs = self.model.predict_proba(self.x_test)[:,1]
        fpr,tpr,_ = roc_curve(self.y_test,y_probs)

        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()


