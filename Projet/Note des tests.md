# Parcours de tests effectués sur le projet pour avoir nos models de prédiction.

Pour notre nos premiers tests, on est parti du modèle fait en fin de TP3a qui est un Word2Vec sur un dataset pré-entrainé avec une Regression Linéaire pour faire la classification sur Chirac/Mitterrand.  
On a ensuite remplacer le Word2Vec par une TF-IDF et on a eu de meilleurs résultats que le Word2Vec.  
Puis on a testé avec une SVM Linéaire au lieu d'une Regression Linéaire et on a eu de meilleurs résultats que la Regression Linéaire.

On a essayé de faire varier les paramètre min_df et max_df de la TF-IDF pour voir si on pouvait améliorer les résultats, mais on n'a pas eu de meilleurs résultats que les paramètres par défaut.

Puis on a remplacé la SVM Linéaire d'autre classificateurs.

| tf-idf + | SVM linéaire | Random Forest | SVM (rbf) | SVM (poly) |
| :------- |:-------:| :------:| :------:| :------:|
| accuracy | 90.229% | 87.329% | 89.767% | 87.381% |
| f1_score | 94.550% | 93.216% | 94.419% | 93.236% |
| auc      | 69.886% | 51.022% | 62.143% | 51.541% |

