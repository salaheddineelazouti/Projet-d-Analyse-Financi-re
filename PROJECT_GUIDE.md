# Guide d'Utilisation du Projet d'Analyse Financière

## Vue d'ensemble

Ce projet réalise une analyse financière approfondie enrichie par des techniques de Data Science et de Machine Learning. Il utilise un dataset contenant 200 indicateurs financiers d'entreprises américaines entre 2014 et 2018.

## Structure du Projet

Le projet est organisé en 5 notebooks Jupyter qui analysent progressivement les données financières:

1. **1_data_loading_exploration.ipynb**:
   - Chargement des données
   - Exploration initiale
   - Nettoyage des données

2. **2_financial_ratios_visualization.ipynb**:
   - Calcul des ratios financiers (ROA, ROE, marge nette, etc.)
   - Visualisation des performances par secteur
   - Analyse des corrélations entre variables financières

3. **3_clustering_and_pca.ipynb**:
   - Réduction de dimension par PCA (Analyse en Composantes Principales)
   - Classification des entreprises en clusters par K-Means
   - Visualisation et interprétation des clusters

4. **4_predictive_modeling.ipynb**:
   - Modèles d'arbres de décision et Random Forest
   - Régressions Ridge et Lasso
   - Identification des facteurs clés de performance

5. **5_conclusions_and_interpretations.ipynb**:
   - Synthèse des résultats des analyses précédentes
   - Interprétation économique des résultats
   - Recommandations stratégiques

## Installation des dépendances

Avant d'exécuter les notebooks, installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## Exécution des notebooks

Les notebooks doivent être exécutés dans l'ordre (1 à 5), car chaque notebook génère des fichiers de données utilisés par les notebooks suivants.

1. Démarrez Jupyter Notebook :
   ```bash
   jupyter notebook
   ```

2. Ouvrez et exécutez chaque notebook séquentiellement.

## Fichiers de données intermédiaires

Le projet génère plusieurs fichiers CSV intermédiaires :
- `cleaned_financial_data.csv` : Données nettoyées
- `financial_data_with_ratios.csv` : Données avec ratios financiers calculés
- `financial_data_with_clusters.csv` : Données avec clusters assignés

## Fonctionnalités principales

- **Exploration des données (EDA)** : Analyse statistique et visualisation des distributions
- **Calcul des ratios financiers** : ROA, ROE, marges, ratios d'endettement, etc.
- **Classification** : Regroupement des entreprises selon leurs profils financiers
- **Analyse en Composantes Principales** : Identification des variables explicatives majeures
- **Modèles prédictifs** : Prédiction de variables financières clés
- **Interprétation économique** : Analyse des facteurs de réussite financière

## Adaptation du projet

Pour adapter ce projet à d'autres datasets financiers :
1. Modifiez le chemin de chargement des données dans le premier notebook
2. Ajustez les noms des colonnes si nécessaire dans les fonctions de calcul des ratios
3. Modifiez la variable cible dans le notebook 4 selon vos besoins d'analyse

## Notes importantes pour le travail en groupe

1. **Répartition des tâches** : Chaque membre du groupe peut se concentrer sur un notebook spécifique
2. **Interprétation collaborative** : Le notebook 5 bénéficie particulièrement d'une analyse collaborative
3. **Documentation** : Documentez vos observations et interprétations dans les notebooks
4. **Livrable final** : Compilez les résultats clés dans un rapport ou une présentation

## Ressources additionnelles

- Documentation de pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- Documentation de scikit-learn: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
- Documentation de matplotlib: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- Documentation de seaborn: [https://seaborn.pydata.org/tutorial.html](https://seaborn.pydata.org/tutorial.html)
