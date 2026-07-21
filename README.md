# Traitement Automatique du Langage (TAL / NLP) — Master 1

Projet d'analyse de sentiments et travaux pratiques réalisés dans le cadre du Master 1.

## Stack Technique

* **NLP & ML :** `PyTorch`, `Hugging Face (transformers)`, `scikit-learn`, `python-crfsuite`, `Gensim` (Word2Vec / FastText)
* **Data & Visu :** `numpy`, `matplotlib`, `wordcloud`
* **Workflow :** Python, Jupyter, LaTeX

## Structure du dépôt

```text
.
├── Projet/                          # Projet final : analyse de sentiments (films & discours)
│   ├── Analyse du projet.ipynb      # EDA (Loi de Zipf, Nuages de mots)
│   ├── notebook_sentiment_transformers_explique.ipynb # Fine-tuning Transformers
│   ├── preprocessing.py / load_data.py # Pipeline de données
│   ├── optimal_Gridsearch.py        # Optimisation d'hyperparamètres
│   └── latex/                       # Rapport de synthèse (rapport.pdf)
│
├── TP1 - Bag of Words/              # Classification TF-IDF / Naive Bayes / RegLog
├── TP2 - Part-Of-Speech/            # Séquences (CRF) & Clustering
├── TP3 - Neural Embeddings/         # Word2Vec, FastText & Intégration CRF
└── TP4 - Transformer/               # RNNs & Fine-tuning Transformers