import numpy as np
import matplotlib.pyplot as plt
from load_data import load_pres

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC


data_path = "../datasets/"
fname = data_path + "corpus.tache1.learn.utf8"
alltxts, y = load_pres(fname)

X = TfidfVectorizer().fit_transform(alltxts)

RANDOM_STATE = 42
N_SPLIT = 10

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=split_ratio, random_state=random_state
# )


clf = SVC(kernel='poly', class_weight='balanced', random_state=RANDOM_STATE)#.fit(X_train, y_train)
# pred = svm_clf.predict(X_test)

# accuracy = accuracy_score(y_test, pred)
# f1_score_ = f1_score(y_test, pred)
# auc = roc_auc_score(y_test, pred)

# print(f"{accuracy = :.3%}")
# print(f"f1_score = {f1_score_:.3%}")
# print(f"{auc = :.3%}")
# print(f"{accuracy = :.3%}, f1_score = {f1_score_:.3%}, {auc = :.3%}")

scoring = ['precision', 'recall','f1', 'roc_auc']
cv_strategy = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(clf, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

# --- Affichage des résultats ---
print(f"Résultats sur {N_SPLIT}-Fold Cross-Validation :")
print("-" * 40)
print(f"| Précision | {cv_results['test_precision'].mean():.3%} (+/- {cv_results['test_precision'].std():.3%}) |")
print(f"| Rappel | {cv_results['test_recall'].mean():.3%} (+/- {cv_results['test_recall'].std():.3%}) |")
print(f"| F1-score | {cv_results['test_f1'].mean():.3%} (+/- {cv_results['test_f1'].std():.3%}) |")
print(f"| ROC AUC  | {cv_results['test_roc_auc'].mean():.3%} (+/- {cv_results['test_roc_auc'].std():.3%}) |")
