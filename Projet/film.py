import numpy as np
import matplotlib.pyplot as plt
from load_data import load_movies

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data_path = "../datasets/"
path = data_path + "movies1000/"
alltxts, y = load_movies(path)

split_ratio = 0.8
random_state = 42

tfid = True
if tfid:
    X = TfidfVectorizer().fit_transform(alltxts)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=split_ratio, random_state=random_state
)

clf = LinearSVC(random_state=random_state).fit(X_train, y_train)
pred = clf.predict(X_test)



# Calcul et affichage des performances du modèle
accuracy = accuracy_score(y_test, pred)
f1_score_ = f1_score(y_test, pred)
auc = roc_auc_score(y_test, pred)

# print(f"{accuracy = :.3%}")
# print(f"f1_score = {f1_score_:.3%}")
# print(f"{auc = :.3%}")
print(f"{accuracy = :.3%}, f1_score = {f1_score_:.3%}, {auc = :.3%}")
    