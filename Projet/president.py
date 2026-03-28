import numpy as np
import matplotlib.pyplot as plt
from load_data import load_pres
from preprocessing import preprocess

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


data_path = "../datasets/"
fname = data_path + "corpus.tache1.learn.utf8"
alltxts, y = load_pres(fname)

alltxts = preprocess(texts=alltxts)
X = TfidfVectorizer(ngram_range=(1, 2), strip_accents=None, lowercase=False, preprocessor=None).fit_transform(alltxts)

RANDOM_STATE = 42
N_SPLIT = 10

# clf = LinearSVC(class_weight='balanced', C= 0.95, random_state=RANDOM_STATE)

# scoring = ['f1', 'roc_auc', 'average_precision']
# cv_strategy = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
# cv_results = cross_validate(clf, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

# # --- Affichage des résultats ---
# print(f"Résultats sur {N_SPLIT}-Fold Cross-Validation :")
# print("-" * 40)
# print(f"| F1-score | {cv_results['test_f1'].mean():.3%} (+/- {cv_results['test_f1'].std():.3%}) |")
# print(f"| ROC AUC  | {cv_results['test_roc_auc'].mean():.3%} (+/- {cv_results['test_roc_auc'].std():.3%}) |")
# print(f"| Avg Precision | {cv_results['test_average_precision'].mean():.3%} (+/- {cv_results['test_average_precision'].std():.3%}) |")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=RANDOM_STATE
)


clf = LinearSVC(C= 0.95, random_state=RANDOM_STATE).fit(X_train, y_train)
pred = clf.predict(X_test)

average_precision = average_precision_score(y_test, pred)
f1_score_ = f1_score(y_test, pred)
auc = roc_auc_score(y_test, pred)

print(f"f1_score = {f1_score_:.3%}")
print(f"{auc = :.3%}")
print(f"{average_precision = :.3%}")
print(f"{average_precision = :.3%}, f1_score = {f1_score_:.3%}, {auc = :.3%}")