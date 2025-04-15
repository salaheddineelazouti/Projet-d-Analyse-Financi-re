import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Créer le dossier figures s'il n'existe pas
figures_dir = 'figures'
os.makedirs(figures_dir, exist_ok=True)

# Configuration des styles de visualisation
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Charger les données
print("Chargement des données...")
try:
    # Essayer d'abord le fichier avec les ratios
    df = pd.read_csv('financial_data_with_ratios.csv')
    print("Fichier financial_data_with_ratios.csv chargé avec succès")
except:
    try:
        # Sinon essayer le fichier nettoyé
        df = pd.read_csv('cleaned_financial_data.csv')
        print("Fichier cleaned_financial_data.csv chargé avec succès")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        exit(1)

# Vérifier si 'Sector' est dans les colonnes
if 'Sector' not in df.columns:
    # Essayons de trouver la colonne du secteur
    potential_sector_cols = [col for col in df.columns if 'sector' in col.lower() or 'industry' in col.lower()]
    if potential_sector_cols:
        # Utiliser la première colonne trouvée
        df['Sector'] = df[potential_sector_cols[0]]
    else:
        # Si pas de colonne secteur, regarder la dernière colonne
        df['Sector'] = df.iloc[:, -1]

# Visualisation 1: Distribution des entreprises par secteur
print("Génération de la visualisation des entreprises par secteur...")
plt.figure(figsize=(14, 10))
sector_counts = df['Sector'].value_counts()
sns.barplot(x=sector_counts.values, y=sector_counts.index, palette='viridis')
plt.title('Nombre d\'entreprises par secteur', fontsize=18)
plt.xlabel('Nombre d\'entreprises', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'companies_by_sector.png'), dpi=300)
plt.close()

# Visualisation 2: Distribution des principales métriques financières
print("Génération des distributions des métriques financières...")
# Identifier les colonnes numériques
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Sélectionner des métriques financières importantes (si nous pouvons les identifier)
key_metrics = []

# Chercher des colonnes qui pourraient contenir des métriques financières importantes
metric_keywords = ['revenue', 'profit', 'income', 'assets', 'debt', 'equity', 'margin', 'ratio']
for keyword in metric_keywords:
    for col in numeric_cols:
        if keyword.lower() in col.lower():
            key_metrics.append(col)
            if len(key_metrics) >= 5:  # Limiter à 5 métriques pour l'affichage
                break
    if len(key_metrics) >= 5:
        break

# Si nous n'avons pas trouvé assez de métriques, prendre les 5 premières colonnes numériques
if len(key_metrics) < 5:
    remaining = 5 - len(key_metrics)
    for col in numeric_cols:
        if col not in key_metrics:
            key_metrics.append(col)
            remaining -= 1
            if remaining == 0:
                break

# Créer des histogrammes pour chaque métrique clé
fig, axes = plt.subplots(len(key_metrics), 1, figsize=(14, 5*len(key_metrics)))
if len(key_metrics) == 1:
    axes = [axes]  # Assurer que axes est toujours une liste
    
for i, metric in enumerate(key_metrics):
    sns.histplot(df[metric].dropna(), kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution de {metric}', fontsize=16)
    axes[i].set_xlabel(metric, fontsize=12)
    axes[i].set_ylabel('Fréquence', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'financial_metrics_distribution.png'), dpi=300)
plt.close()

# Visualisation 3: Corrélation entre les métriques financières
print("Génération de la matrice de corrélation...")
# Sélectionner un sous-ensemble de colonnes pour la corrélation (pour éviter une matrice trop grande)
if len(key_metrics) >= 10:
    corr_metrics = key_metrics[:10]
else:
    corr_metrics = key_metrics

# Calculer la matrice de corrélation
corr_matrix = df[corr_metrics].corr()

# Visualiser la matrice de corrélation
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Matrice de corrélation des métriques financières clés', fontsize=18)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'correlation_matrix.png'), dpi=300)
plt.close()

# Visualisation 4: Identifier les ratios financiers importants (si disponibles)
print("Recherche et visualisation des ratios financiers...")
# Mots-clés pour identifier les ratios
ratio_keywords = ['ROA', 'ROE', 'profit_margin', 'debt_ratio', 'liquidity']
ratio_cols = []

# Chercher des colonnes qui pourraient contenir des ratios financiers
for keyword in ratio_keywords:
    for col in numeric_cols:
        if keyword.lower() in col.lower():
            ratio_cols.append(col)
            break

# Si nous n'avons pas trouvé de ratios, essayer de les créer nous-mêmes
if not ratio_cols and len(key_metrics) >= 2:
    # On pourrait essayer de créer des ratios simples à partir des métriques identifiées
    # Par exemple: métrique1 / métrique2
    # Mais pour l'instant, utilisons ce que nous avons
    ratio_cols = key_metrics[:2]

if ratio_cols:
    # Visualiser les ratios par secteur
    for ratio in ratio_cols[:3]:  # Limiter à 3 ratios
        plt.figure(figsize=(14, 10))
        sns.boxplot(x='Sector', y=ratio, data=df, palette='Set3')
        plt.title(f'{ratio} par secteur', fontsize=18)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'{ratio}_by_sector.png'), dpi=300)
        plt.close()

# Visualisation 5: Essayer de visualiser les outliers
print("Visualisation des outliers...")
# Sélectionner quelques métriques pour la détection d'outliers
outlier_metrics = key_metrics[:3]  # Utiliser jusqu'à 3 métriques

for metric in outlier_metrics:
    plt.figure(figsize=(12, 8))
    sns.boxplot(y=df[metric])
    plt.title(f'Détection d\'outliers pour {metric}', fontsize=18)
    plt.ylabel(metric, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'outliers_{metric}.png'), dpi=300)
    plt.close()

print(f"Toutes les visualisations ont été générées et sauvegardées dans le dossier '{figures_dir}'")
