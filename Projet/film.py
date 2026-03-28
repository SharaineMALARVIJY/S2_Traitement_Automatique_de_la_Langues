import numpy as np
import matplotlib.pyplot as plt
from load_data import load_movies, load_movies_test
from preprocessing import preprocess

from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC, NuSVC
# from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

data_path = "../datasets/"
path = data_path + "movies1000/"
alltxts, y = load_movies(path)
alltxts_test = load_movies_test("../datasets_test_projet/testSentiment.txt")
RANDOM_STATE = 42
N_SPLIT = 5

alltxts = preprocess(texts=alltxts)
vectorizer = TfidfVectorizer(min_df=0.005, ngram_range=(1, 2), sublinear_tf=True, strip_accents=None, lowercase=False, preprocessor=None)
X = vectorizer.fit_transform(alltxts)

alltxts_test = preprocess(texts=alltxts_test)
X_test = vectorizer.transform(alltxts_test)
# clf = LinearSVC(random_state=RANDOM_STATE)#.fit(X_train, y_train)
# # pred = clf.predict(X_test)

# # Calcul des performances du modèle
# scoring = ['accuracy', 'f1', 'precision', 'recall']
# cv_strategy = KFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)
# cv_results = cross_validate(clf, X, y, cv=cv_strategy, scoring=scoring, n_jobs=-1)

# # --- Affichage des résultats ---
# print(f"Résultats sur {N_SPLIT}-Fold Cross-Validation :")
# print("-" * 40)
# print(f"| Accuracy | {cv_results['test_accuracy'].mean():.3%} (+/- {cv_results['test_accuracy'].std():.3%}) |")
# print(f"| F1-score | {cv_results['test_f1'].mean():.3%} (+/- {cv_results['test_f1'].std():.3%}) |")
# print(f"| Précision | {cv_results['test_precision'].mean():.3%} (+/- {cv_results['test_precision'].std():.3%}) |")
# print(f"| Rappel | {cv_results['test_recall'].mean():.3%} (+/- {cv_results['test_recall'].std():.3%}) |")


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=0.8, random_state=RANDOM_STATE
# )

# clf = SVC(kernel='rbf', verbose=True, random_state=RANDOM_STATE).fit(X_train, y_train)
# pred = clf.predict(X_test)

# accuracy = accuracy_score(y_test, pred)
# f1_score_ = f1_score(y_test, pred)
# auc = roc_auc_score(y_test, pred)

# pred = clf.predict(X_train)
# accuracy_train = accuracy_score(y_train, pred)
# f1_score__train = f1_score(y_train, pred)
# auc_train = roc_auc_score(y_train, pred)

# print(f"{accuracy_train = :.3%}")
# print(f"f1_score_train = {f1_score__train:.3%}")
# print(f"{auc_train = :.3%}")

# print(f"{accuracy = :.3%}")
# print(f"f1_score = {f1_score_:.3%}")
# print(f"{auc = :.3%}")
# # print(f"{accuracy = :.3%}, f1_score = {f1_score_:.3%}, {auc = :.3%}")

clf = LinearSVC(random_state=RANDOM_STATE).fit(X, y)

pred = clf.predict(X_test)
arr = np.where(pred == 1, 'P', 'N')

# Sauvegarder le résultat dans un fichier
import glob
m = len(glob.glob('csv/submission-movie-*.csv')) + 1
np.savetxt(f"csv/submission-movie-{m}.csv", arr, fmt="%s")