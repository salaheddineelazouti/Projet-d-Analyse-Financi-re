# Analyse Financière avec Data Science & Machine Learning
![École Centrale Casablanca](figures/logo-ecc.png)

[![LaTeX](https://img.shields.io/badge/Rapport-LaTeX-blue.svg)](rapport.pdf)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-green.svg)](https://www.python.org/downloads/release/python-390/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Projet GE (Génie Entreprises) - École Centrale Casablanca*

## Aperçu
Ce projet réalise une analyse financière enrichie en utilisant des techniques de Data Science et Machine Learning pour :
- Identifier les tendances cachées dans les données financières
- Classifier les entreprises selon des critères financiers
- Prédire des variables financières
- Offrir une valeur éducative en combinant des compétences en finance et en science des données

## Jeu de données
Le jeu de données contient 200 indicateurs financiers d'actions américaines de 2014 à 2018, incluant :
- Nom et ticker des entreprises
- Informations sectorielles
- Métriques de revenus et de profits
- Informations sur les actifs et passifs
- Indicateurs de performance de marché

## Structure du projet
```
financial_analysis/
├── notebooks/                      # Jupyter notebooks pour l'analyse
│   ├── 1_data_loading_exploration.ipynb
│   ├── 2_financial_ratios_visualization.ipynb
│   ├── 3_clustering_and_pca.ipynb
│   ├── 4_predictive_modeling.ipynb
│   └── 5_conclusions_and_interpretations.ipynb
├── data/                           # Données brutes et traitées
├── figures/                        # Visualisations générées
├── rapport.tex                     # Rapport LaTeX
├── rapport.pdf                     # Rapport compilé en PDF
├── requirements.txt                # Dépendances Python
├── execute_notebooks.py            # Script d'exécution automatique
└── generate_visualizations_improved.py  # Génération des visualisations
```

## Méthodologie
1. **Analyse Exploratoire des Données (EDA)**
   - Nettoyage et traitement des données
   - Analyse de distribution des variables clés
   - Calcul des ratios financiers (ROA, ROE, marge nette, etc.)
   - Visualisations de corrélation

2. **Applications de Machine Learning**
   - Classification (K-Means, ACP) pour le profilage d'entreprises
   - Arbres de décision/Random Forest pour identification des facteurs de performance
   - Modèles de régression pour prédiction de variables financières
   - Analyse en Composantes Principales pour réduction de dimensionnalité

3. **Interprétation Économique**
   - Analyse des facteurs de succès
   - Identification des profils d'entreprise
   - Relations entre secteurs et performance

## Installation et utilisation

### Prérequis
- Python 3.9+
- LaTeX (pour générer le rapport)

### Installation
1. Cloner le dépôt
   ```bash
   git clone [https://github.com/votre-nom/financial-analysis.git](https://github.com/salaheddineelazouti/Projet-d-Analyse-Financi-re)
   cd financial-analysis
   ```

2. Créer un environnement virtuel (recommandé)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installer les dépendances
   ```bash
   pip install -r requirements.txt
   ```

### Utilisation
1. Explorer les notebooks individuellement
   ```bash
   jupyter notebook notebooks/
   ```

2. Exécuter tous les notebooks séquentiellement
   ```bash
   python execute_notebooks.py
   ```

3. Générer les visualisations
   ```bash
   python generate_visualizations_improved.py
   ```

4. Compiler le rapport LaTeX (nécessite une installation LaTeX)
   ```bash
   cd rapport
   pdflatex rapport.tex
   pdflatex rapport.tex  # Seconde exécution pour les références
   ```

## Résultats clés
- Identification de trois clusters d'entreprises avec des profils financiers distincts
- Corrélation significative entre l'efficacité opérationnelle et la performance boursière
- Impact sectoriel variable sur les indicateurs financiers fondamentaux
- Modèles prédictifs avec une précision de plus de 80% pour certaines métriques financières

## Équipe du projet
| Nom | Rôle dans ce projet|
|------|------|
| Salah Eddine EL AZOUTI | Chef de projet & Data Scientist |
| Salma Saaidi | Analyste financière |
| Anas EL HAYEL | Data Engineer |
| Walid EL BOUCHTI | ML Engineer |
| Aasma ouamalich | Financial Analyst |

Ces rôles sont répartis  pour assurer le bon deroulement de projet
## Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## Liens
- [École Centrale Casablanca](https://www.centrale-casablanca.ma/)
- [Consulter le rapport en ligne](https://github.com/salaheddineelazouti/Projet-d-Analyse-Financi-re/blob/master/rapport%20GE.pdf)

  
