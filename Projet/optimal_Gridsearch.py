from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from load_data import load_pres, load_movies
from preprocessing import preprocess
#from nltk.corpus import stopwords

#stopwords_en = list(stopwords.words('english'))
#stopwords_fr = list(stopwords.words('french'))

data_path = "../datasets/"
fname = data_path + "corpus.tache1.learn.utf8"
alltxts, y = load_pres(fname)

# path = data_path + "movies1000/"
# alltxts, y = load_movies(path)

RANDOM_STATE = 42
N_SPLIT = 10

alltxt = preprocess(texts=alltxts)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(sublinear_tf=True)),
    ('clf',   LinearSVC(random_state=RANDOM_STATE)),
    # ('clf', RandomForestClassifier(random_state=RANDOM_STATE)),
    # ('clf', MultinomialNB()),
    # ('clf', LogisticRegression(random_state=RANDOM_STATE)),
])

param_grid = {
    'tfidf__ngram_range':  [(1, 2)],
    'tfidf__max_df':       [0.85, 0.95, 1.0],
    'tfidf__min_df':       [1, 2, 3, 5, 10],
    'tfidf__max_features': [None, 37_000],
    'tfidf__smooth_idf':   [False],
    'tfidf__use_idf':      [True, False], 
    'tfidf__sublinear_tf': [True],
    'tfidf__norm':         ['l2', 'l1', None],
    'tfidf__analyzer':     ['word', 'char_wb'],
    #"tfidf__stop_words": [None, stopwords_fr],
    'clf__C': [1, 0.9, 0.95],
    #'clf__kernel': ['linear', 'rbf', 'poly'],
}
cv_strategy = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_strategy,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
    refit=True,
)

grid_search.fit(alltxts, y)

print("Meilleurs paramètres :")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"  Meilleur F1 (CV interne) : {grid_search.best_score_:.3%}")

#scoring = ['accuracy', 'f1', 'roc_auc']
scoring = ['f1', 'roc_auc', 'average_precision']

cv_results = cross_validate(
    grid_search.best_estimator_,
    alltxts, y,
    cv=cv_strategy,
    scoring=scoring,
    n_jobs=-1,
)

print(f"Résultats sur {N_SPLIT}-Fold Cross-Validation :")
print(f"| F1-score | {cv_results['test_f1'].mean():.3%} (+/- {cv_results['test_f1'].std():.3%}) |")
print(f"| ROC AUC  | {cv_results['test_roc_auc'].mean():.3%} (+/- {cv_results['test_roc_auc'].std():.3%}) |")
print(f"| Avg Precision | {cv_results['test_average_precision'].mean():.3%} (+/- {cv_results['test_average_precision'].std():.3%}) |")