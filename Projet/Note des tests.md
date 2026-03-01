# Informations sur les datasets 
## Chirac/Mitterrand
Nombre de phrases : 57413  
Chirac : 49890 (86,90%)  
Mitterrand : 7523 (13,10%)   

## Commentaire de Film 
Nombre de commentaires : 2000  
Positifs : 1000 (50%)  
Negatifs : 1000 (50%)  


# Parcours de tests effectués sur le projet pour avoir nos models de prédiction.

Pour notre nos premiers tests, on est parti du modèle fait en fin de TP3a qui est un Word2Vec sur un dataset pré-entrainé avec une Regression Linéaire pour faire la classification sur Chirac/Mitterrand.  
On a ensuite remplacer le Word2Vec par une TF-IDF et on a eu de meilleurs résultats que le Word2Vec.  
Puis on a testé avec une SVM Linéaire au lieu d'une Regression Linéaire et on a eu de meilleurs résultats que la Regression Linéaire.

On a essayé de faire varier les paramètre min_df et max_df de la TF-IDF pour voir si on pouvait améliorer les résultats, mais on n'a pas eu de meilleurs résultats que les paramètres par défaut.

Puis on a remplacé la SVM Linéaire d'autre classificateurs.


## Chirac/Mitterrand

(Pour ce tableau de tests, je n'ai pas pris en compte le fait que le dataset n'est pas équilibré dans les paramètres des classificateurs)

| tf-idf + | SVM linéaire | Random Forest | SVM (rbf) | SVM (poly) |
| :------- |:-------:| :------:| :------:| :------:|
| accuracy | 90.229% | 87.329% | 89.767% | 87.381% |
| f1_score | 94.550% | 93.216% | 94.419% | 93.236% |
| ROC AUC  | 69.886% | 51.022% | 62.143% | 51.541% |

# Test de tf-idf + SVM linéaire sur les deux datasets

## Commentaire de Film 

```python
from sklearn.svm import LinearSVC, SVC
clf = LinearSVC()
clf = SVC(kernel="linear")   
```
Etonnament, le deux classificateurs SVM linéaire ne donnent pas les mêmes résultats.


| tf-idf + | SVM linéaire | SVM (kernel = linear) |
| :------- |:-------:| :------:|
| accuracy | 83.500% | 85.250% |
| f1_score | 83.981% | 85.783% |
| ROC AUC  | 83.487% | 85.233% |
| Temps d'execution | 3s | 7s |

J'ai refait les tests avec une stratified k-fold cross validation pour avoir des résultats plus fiables, et les résultats sont plus cohérents que ceux obtenus avec un simple train/test split.

Résultats avec 5 folds :

| tf-idf + | SVM linéaire | SVM (kernel = linear) |
| :------- |:-------:| :------:|
| Accuracy | 85.650% (+/- 1.586%) | 85.350% (+/- 0.943%) |
| F1-score | 85.731% (+/- 1.546%) | 85.403% (+/- 0.873%) |
| ROC AUC  | 93.182% (+/- 0.612%) | 92.926% (+/- 0.672%) |
| Temps d'execution | 5s | 15s |

## Chirac/Mitterrand

Je refait les tests avec une stratified k-fold cross validation.
Ici la k-fold cross validation a l'avantage de pouvoir réduire le probleme du dataset non équilibré, et de paralleliser les calculs pour réduire le temps d'execution.

```python
colonne1 = LinearSVC()
colonne2 = LinearSVC(class_weight='balanced')
colonne3 = SVC(kernel='linear', class_weight='balanced')
colonne4 = SVC(kernel='rbf', class_weight='balanced')
colonne5 = SVC(kernel='poly', class_weight='balanced')
```
Contrairement au premier test, j'ai pris en compte le déséquilibre du dataset en utilisant le paramètre class_weight='balanced' pour les SVM linéaire, rbf et poly.

Résultats avec 10 folds :  

| tf-idf + | SVM linéaire (without balanced) | SVM linéaire | SVM (linear) | SVM (rbf) | SVM (poly) | 
| :------- |:-------:| :------:| :------:| :------:| :------:|
| Précision | 91.814% (+/- 0.255%) | 94.433% (+/- 0.213%) | 95.025% (+/- 0.275%) | 92.725% (+/- 0.294%) | 87.818% (+/- 0.123%) |
| Rappel | 97.563% (+/- 0.182%) | 88.779% (+/- 0.428%) | 86.560% (+/- 0.344%) | 96.360% (+/- 0.262%) | 99.555% (+/- 0.109%) |
| F1-score | 94.601% (+/- 0.195%) | 91.519% (+/- 0.266%) | 90.595% (+/- 0.271%) | 94.507% (+/- 0.248%) | 93.319% (+/- 0.069%) |
| ROC AUC  | 87.679% (+/- 0.697%) | 86.870% (+/- 0.646%) | 87.331% (+/- 0.660%) | 88.952% (+/- 0.646%) |  87.527% (+/- 0.670%) |
| Temps d'execution | 6s | 6s | 13m 22s | 23m 22s | 44m 48s |

La fonction LinearSVC est bien plus optimisée sur les grands datasets que la fonction SVC. 
