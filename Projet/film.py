import numpy as np
import matplotlib.pyplot as plt
from load_data import load_movies

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC, NuSVC
# from sklearn.ensemble import RandomForestClassifier

data_path = "../datasets/"
path = data_path + "movies1000/"
alltxts, y = load_movies(path)

RANDOM_STATE = 42
N_SPLIT = 6

X = TfidfVectorizer().fit_transform(alltxts)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=split_ratio, random_state=RANDOM_STATE
# )

clf = SVC(kernel='linear', random_state=RANDOM_STATE)#.fit(X_train, y_train)
# pred = clf.predict(X_test)

# Calcul des performances du modèle
scoring = ['accuracy', 'f1', 'roc_auc']
cv_strategy = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
cv_results = cross_validate(clf, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

# --- Affichage des résultats ---
print(f"Résultats sur {N_SPLIT}-Fold Cross-Validation :")
print("-" * 40)
print(f"| Accuracy | {cv_results['test_accuracy'].mean():.3%} (+/- {cv_results['test_accuracy'].std():.3%}) |")
print(f"| F1-score | {cv_results['test_f1'].mean():.3%} (+/- {cv_results['test_f1'].std():.3%}) |")
print(f"| ROC AUC  | {cv_results['test_roc_auc'].mean():.3%} (+/- {cv_results['test_roc_auc'].std():.3%}) |")
