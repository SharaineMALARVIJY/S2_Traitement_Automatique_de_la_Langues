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

# Soumision 1
clf = LinearSVC().fit(X, y)

# Soumision 3
# clf = LogisticRegression().fit(X, y)


pred = clf.predict(X_test)
arr = np.where(pred == 1, 'P', 'N')

# # Sauvegarder le résultat dans un fichier
# import glob
# m = len(glob.glob('csv/submission-movie-*.csv')) + 1
# np.savetxt(f"csv/submission-movie-{m}.csv", arr, fmt="%s")