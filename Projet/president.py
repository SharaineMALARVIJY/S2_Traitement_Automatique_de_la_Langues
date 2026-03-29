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


clf = LinearSVC(class_weight='balanced', C= 0.95, random_state=RANDOM_STATE).fit(X, y)
pred = clf.predict(X_test)

#condition, resultat_if, resultat_else
arr = np.where(pred == -1, 1, 0)

# Sauvegarder le résultat dans un fichier
# import glob
# n = len(glob.glob('csv/submission-pres-*.csv')) + 1
# np.savetxt(f"csv/submission-pres-{n}.csv", arr, fmt="%s")



### Post-praitement

from scipy.ndimage import gaussian_filter

nb_graph_phrase = 1000

plt.figure()
plt.plot(list(range(len(y[0:nb_graph_phrase]))), y[0:nb_graph_phrase])
plt.xlabel("Phrases")
plt.ylabel("Locuteur")
plt.show()


# probs = proba_sentiment(alltxts[:nb_graph_phrase], model, tokenizer, device, maxL)
    
plt.figure()
plt.plot(list(range(len(y[0:nb_graph_phrase]))), arr[0:nb_graph_phrase])
plt.xlabel("Phrases")
plt.ylabel("Locuteur")
plt.show()

plt.figure()
plt.plot(list(range(len(y[0:nb_graph_phrase]))), gaussian_filter(input=arr[0:nb_graph_phrase], sigma=1.25))
plt.xlabel("Phrases")
plt.ylabel("Locuteur")
plt.show()