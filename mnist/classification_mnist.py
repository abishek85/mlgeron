#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:24:57 2019
MNIST classification
@author: abishekk
"""

import numpy as np
import os
from sklearn.datasets import fetch_openml

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# helper functions
def plot_digit(digit_row):
    img = digit_row.reshape(28, 28)
    plt.imshow(img, cmap = mpl.cm.binary, interpolation="nearest")
    plt.axis("off")

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="upper left", fontsize=16)
    plt.ylim([0, 1])
    
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

#%% Data import
mnist_data = fetch_openml('mnist_784', version=1, cache=True)
mnist_data.target = mnist_data.target.astype(np.int8)

#%% Data viz
X, y = mnist_data.data, mnist_data.target

print(X)
print(X.shape)

print(y)
print(y.shape)

plot_digit(X[42142])
print(y[42142])

#%% Test-train split and shuffle

num_train = 60000

X_train, X_test = X[:num_train], X[num_train:]
y_train, y_test = y[:num_train], y[num_train:]

shuffle_index = np.random.permutation(num_train)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

#%% Binary classifier
y_train_4 = (y_train == 4)
y_test_4 = (y_test == 4)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=20, tol=-np.infty, random_state=42)

sgd_clf.fit(X_train,y_train_4)

#%% Classifier accuracy
from sklearn.model_selection import cross_val_score

scores = cross_val_score(sgd_clf, X_train, y_train_4, cv=3, scoring="accuracy")
print(scores)


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_4):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_4[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_4[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

#%% Confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_4, cv=3)

print(confusion_matrix(y_train_4, y_train_pred))
print("Precision: ", precision_score(y_train_4, y_train_pred))
print("Recall: ", recall_score(y_train_4, y_train_pred))
print("F1 score: ", f1_score(y_train_4, y_train_pred))

#%% Decision scores to control precision and recall
from sklearn.metrics import  precision_recall_curve, roc_curve, roc_auc_score

y_scores = cross_val_predict(sgd_clf, X_train, y_train_4, cv=3,
                             method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_4, y_scores)
plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-700000, 700000])
plt.show()

falsepos_rate, truepos_rate, thresholds = roc_curve(y_train_4, y_scores)
plt.figure(figsize=(8, 6))
plot_roc_curve(falsepos_rate, truepos_rate)
plt.show()

print("ROC AUC score:", roc_auc_score(y_train_4, y_scores))

#%% Compare SGD against RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=20, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_4, cv=3,
                                    method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] 
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_4,y_scores_forest)

plt.figure(figsize=(8, 6))
plt.plot(falsepos_rate, truepos_rate, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
plt.show()

print("ROC AUC score:", roc_auc_score(y_train_4, y_scores_forest))

#%% Multiclass classifier
# scikit-learn does one-vs-one or one-vs-rest automatically
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([X[42142]]))

print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

#%% OnevsOne classifier
from sklearn.multiclass import OneVsOneClassifier

ovo_clf = OneVsOneClassifier(SGDClassifier(max_iter=5, tol=-np.infty, 
                                           random_state=42))
ovo_clf.fit(X_train, y_train)
print(len(ovo_clf.estimators_))
print(ovo_clf.predict([X[42142]]))

#%% Random Forest multiclass

forest_clf.fit(X_train, y_train)
print(forest_clf.predict([X[42142]]))
print(forest_clf.predict_proba([X[42142]]))

#%% ErrorAnalysis

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)

plt.matshow(conf_mx)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)

plt.matshow(norm_conf_mx)
plt.show()