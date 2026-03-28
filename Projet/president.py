import numpy as np
import matplotlib.pyplot as plt
from load_data import load_pres, load_pres_test
from preprocessing import preprocess

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression


data_path = "../datasets/"
fname = data_path + "corpus.tache1.learn.utf8"
alltxts, y = load_pres(fname)
alltxts_test = load_pres_test("../datasets_test_projet/corpus.tache1.test.utf8")

alltxts = preprocess(texts=alltxts)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents=None, lowercase=False, preprocessor=None)
X = vectorizer.fit_transform(alltxts)

alltxts_test = preprocess(texts=alltxts_test)
X_test = vectorizer.transform(alltxts_test)

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


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, train_size=0.8, random_state=RANDOM_STATE
# )


# clf = LinearSVC(class_weight='balanced', C= 0.95, random_state=RANDOM_STATE).fit(X_train, y_train)
# pred = clf.predict(X_test)

# average_precision = average_precision_score(y_test, pred)
# f1_score_ = f1_score(y_test, pred)
# auc = roc_auc_score(y_test, pred)

# print(f"f1_score = {f1_score_:.3%}")
# print(f"{auc = :.3%}")
# print(f"{average_precision = :.3%}")
# print(f"{average_precision = :.3%}, f1_score = {f1_score_:.3%}, {auc = :.3%}")






clf = LinearSVC(class_weight='balanced', C= 0.95, random_state=RANDOM_STATE).fit(X, y)
pred = clf.predict(X_test)
#condition, resultat_if, resultat_else
arr = np.where(pred == -1, 1, 0)

# Sauvegarder le résultat dans un fichier
import glob
n = len(glob.glob('csv/submission-pres-*.csv')) + 1
np.savetxt(f"csv/submission-pres-{n}.csv", arr, fmt="%s")






### Graph de post-praitement



# import numpy as np
# import matplotlib.pyplot as plt

# def gaussian_kernel(size, sigma=1.0):
#     """
#     Génère un noyau gaussien normalisé
#     """
#     x = np.arange(size) - size // 2
#     kernel = np.exp(-(x**2) / (2 * sigma**2))
#     kernel = kernel / np.sum(kernel)
#     return kernel


# def gaussian_smoothing(pred, size=5, sigma=1.0):
#     """
#     Applique un lissage gaussien sur les prédictions
    
#     pred : liste ou array de prédictions (-1, 1)
#     size : taille du filtre (impair recommandé)
#     sigma : écart-type de la gaussienne
#     """
#     kernel = gaussian_kernel(size, sigma)
    
#     # Convolution
#     smoothed = np.convolve(pred, kernel, mode='same')
    
#     # Reprojection en classes (-1, 1)
#     return smoothed



# from load_data import load_pres, load_pres_test

# data_path = "../datasets/train/"
# path = data_path + "corpus.tache1.learn.utf8"
# alltxts, y = load_pres(path)
# # alltxts_test = load_movies_test("../datasets/test/testSentiment.txt")



# nb_graph_phrase = 1000

# plt.figure()
# plt.plot(list(range(len(y[0:nb_graph_phrase]))), y[0:nb_graph_phrase])
# plt.xlabel("Phrases")
# plt.ylabel("Locuteur")
# plt.show()


# probs = proba_sentiment(alltxts[:nb_graph_phrase], model, tokenizer, device, maxL)
    
# plt.figure()
# plt.plot(list(range(len(y[0:nb_graph_phrase]))), probs_list[0:nb_graph_phrase], sigma=1.25)
# plt.xlabel("Phrases")
# plt.ylabel("Locuteur")
# plt.show()

# plt.figure()
# plt.plot(list(range(len(y[0:nb_graph_phrase]))), gaussian_smoothing(probs[0:nb_graph_phrase], sigma=1.25))
# plt.xlabel("Phrases")
# plt.ylabel("Locuteur")
# plt.show()