import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
from sklearn import datasets, linear_model
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import random

def read_data(run_num):
    #Source:  Pima-Indian diabetes dataset: https://www.kaggle.com/kumargh/pimaindiansdiabetescsv

    data = pd.read_csv("pima-indians-diabetes.csv", sep=",", header = None)
    X = data.iloc[:,0:8].values
    y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4,stratify=y, random_state=run_num)
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    return x_train_s, x_test_s, y_train, y_test

 
    
def logistic(x_train, x_test, y_train, y_test, l1_ratio=0.5):
    #Source: Scikit Learn. (n.d). Linear Regression Example. https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html 
    penalties= ['l1','l2','elasticnet']
    results = {}

    for pen in penalties:
        if pen == 'elasticnet':
            name = f"elasticnet_{l1_ratio:.2f}"
            model = linear_model.LogisticRegression(penalty=pen, tol=0.01,solver='saga', l1_ratio=l1_ratio)
        else:
            name = pen
            model = linear_model.LogisticRegression(penalty=pen, tol=0.01,solver='saga') #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        model.fit(x_train, y_train)
        
        y_train_preds = model.predict(x_train)
        y_test_preds = model.predict(x_test)
        y_train_probs = model.predict_proba(x_train)[:, 1]
        y_test_probs = model.predict_proba(x_test)[:, 1]
        results[name] = {
            "y_train_pred": y_train_preds,
            "y_test_pred":  y_test_preds,
            "y_train_prob": y_train_probs,
            "y_test_prob":  y_test_probs
        }
        
    return results

def metrics(y_train, y_test, results):
    rows = []
    outputs = {}
    for name, res in results.items():
        ytrp, ytep = res["y_train_pred"], res["y_test_pred"]
        ytrb, yteb = res["y_train_prob"], res["y_test_prob"]
        acc_train = accuracy_score(y_train, ytrp)
        acc_test = accuracy_score(y_test, ytep)
        auc_train = roc_auc_score(y_train, ytrb)
        auc_test = roc_auc_score(y_test, yteb)
        f1_train = f1_score(y_train, ytrp)
        f1_test = f1_score(y_test, ytep)
        fpr, tpr, _ = roc_curve(y_test, yteb)
        precision, recall, _ = precision_recall_curve(y_test, yteb)
    
        outputs[name] = {
            'AUC_test': auc_test, 'F1_test': f1_test,
            'FPR': fpr,'TPR': tpr,
            'Precision': precision,'Recall': recall}
        rows.append([name, acc_train, acc_test, auc_train, auc_test, f1_train, f1_test])

    table = pd.DataFrame(rows,
    columns=["Penalty", "ACC_train", "ACC_test", "AUC_train", "AUC_test", "F1_train", "F1_test"]
    ).set_index("Penalty")
    return table, outputs

def plot_curves(outputs):
    plt.figure()
    for name, m in outputs.items():
        auc_val = m["AUC_test"]
        label = f"{name} (AUC: {auc_val:.2f})"
        plt.plot(m["FPR"], m["TPR"], marker='.',linewidth=1.5, label=label) 
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)    
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('AUC_plot.png')
    plt.show()
    
    plt.figure()
    for name, m in outputs.items():
        label = f"{name} (F1: {m['F1_test']:.2f})"
        plt.plot(m['Recall'], m['Precision'], marker='.', linewidth=1.5, label=label)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Recall_precision_plot.png')
    plt.show()

    
def cross_validating(X, y, l1_ratio=0.5):
    model = linear_model.LogisticRegression(C=1.0, penalty='elasticnet', tol=0.01, solver='saga', l1_ratio=l1_ratio)
    
    scoring = ['roc_auc', 'f1']
    scores = cross_validate(model, X, y, cv=10, scoring=scoring, return_train_score=True)
    
    roc_auc = scores['test_roc_auc']
    f1 = scores['test_f1']
    return roc_auc, f1

def summary_metrics(table):
    # 1) Stack all run tables into one big table
    all_runs = pd.concat(table, axis=0)  # rows just pile up
    # 2) Group by model name (the index) and compute mean & std
    means = all_runs.groupby(all_runs.index).mean()
    stds  = all_runs.groupby(all_runs.index).std()

    # 3) Build a numeric summary with *_mean and *_std columns
    summary = pd.DataFrame(index=means.index)
    for col in means.columns:
        summary[f"{col}_mean"] = means[col]
        summary[f"{col}_std"]  = stds[col]

    # 4) Build a pretty table with "mean ± std" strings
    result_df = pd.DataFrame(index=means.index)
    for col in means.columns:
        result_df[col] = means[col].round(3).astype(str) + " ± " + stds[col].round(3).astype(str)

    return result_df, summary

def main():
    runs = 10
    all_results = []
    for run in range(runs):
        x_train, x_test, y_train, y_test = read_data(run)
        results = logistic(x_train, x_test, y_train, y_test)
        table, output = metrics(y_train, y_test, results)
        all_results.append(table)
        if run == runs - 1:
            plot_curves(output)
    result_df, summary = summary_metrics(all_results)
    print(all_results[-1].round(3))
    print(result_df)
    roc_auc, f1 = cross_validating(x_train, y_train)
    print(f"ROC AUC mean and STD: {np.mean(roc_auc):.3f} ± {np.std(roc_auc):.3f}")
    print(f"F1 mean and STD: {np.mean(f1):.3f} ± {np.std(f1):.3f}")
if __name__ == "__main__":
    main()
