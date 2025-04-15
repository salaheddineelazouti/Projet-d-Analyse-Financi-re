import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# Configuration
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
os.makedirs('figures', exist_ok=True)
print("Generating visualizations for financial analysis report...")

# Chargement des données
try:
    # Essayer d'abord avec les données contenant les ratios
    df = pd.read_csv('financial_data_with_ratios.csv')
    print("Loaded financial_data_with_ratios.csv")
except:
    try:
        # Sinon avec les données nettoyées
        df = pd.read_csv('cleaned_financial_data.csv')
        print("Loaded cleaned_financial_data.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

# Identifier les colonnes importantes
print("Identifying important columns...")
columns = df.columns.tolist()

# Chercher la colonne Secteur
sector_col = None
for col in columns:
    if col.lower() == 'sector' or 'sector' in col.lower():
        sector_col = col
        break

if sector_col:
    print(f"Found sector column: {sector_col}")
else:
    # Si on ne trouve pas de colonne secteur explicite, on utilise la dernière colonne
    sector_col = columns[-1]
    print(f"Using last column as sector: {sector_col}")

# Identifions les colonnes numériques pour les analyses
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 1. VISUALISATION DES ENTREPRISES PAR SECTEUR
print("Generating sector distribution visualization...")
plt.figure(figsize=(14, 10))
sector_counts = df[sector_col].value_counts()
sns.barplot(x=sector_counts.values, y=sector_counts.index, palette='viridis')
plt.title('Nombre d\'entreprises par secteur', fontsize=18)
plt.xlabel('Nombre d\'entreprises', fontsize=14)
plt.tight_layout()
plt.savefig('figures/companies_by_sector.png', dpi=300)
plt.close()

# 2. ANALYSE DES RATIOS FINANCIERS (SI DISPONIBLES)
print("Analyzing financial ratios...")
# Chercher des colonnes qui pourraient être des ratios financiers
ratio_keywords = ['ROA', 'ROE', 'margin', 'ratio', 'turnover']
ratio_cols = []

for keyword in ratio_keywords:
    for col in numeric_cols:
        if keyword.lower() in col.lower():
            ratio_cols.append(col)

# Si on a trouvé des ratios, on les analyse
if ratio_cols:
    print(f"Found {len(ratio_cols)} ratio columns: {ratio_cols}")
    
    # Visualisation des ratios par secteur
    for ratio in ratio_cols[:3]:  # On limite à 3 ratios pour ne pas surcharger
        plt.figure(figsize=(14, 10))
        try:
            sns.boxplot(x=sector_col, y=ratio, data=df)
            plt.title(f'{ratio} par secteur', fontsize=18)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'figures/{ratio}_by_sector.png', dpi=300)
        except Exception as e:
            print(f"Error generating boxplot for {ratio}: {e}")
        plt.close()
else:
    print("No financial ratio columns found")

# 3. ANALYSE DES CORRÉLATIONS
print("Generating correlation matrix...")
# Sélectionner un sous-ensemble de colonnes numériques pour l'analyse de corrélation
# On évite de prendre toutes les colonnes numériques si elles sont trop nombreuses
if len(numeric_cols) > 10:
    # On essaie de prendre des colonnes financières significatives
    important_cols = []
    keywords = ['revenue', 'profit', 'income', 'assets', 'equity', 'margin', 'ROA', 'ROE']
    
    for keyword in keywords:
        for col in numeric_cols:
            if keyword.lower() in col.lower() and col not in important_cols:
                important_cols.append(col)
                if len(important_cols) >= 10:
                    break
        if len(important_cols) >= 10:
            break
    
    # Si on n'a pas assez de colonnes, on complète
    if len(important_cols) < 10:
        for col in numeric_cols:
            if col not in important_cols:
                important_cols.append(col)
                if len(important_cols) >= 10:
                    break
else:
    important_cols = numeric_cols

# Créer la matrice de corrélation
plt.figure(figsize=(14, 12))
try:
    corr_matrix = df[important_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de corrélation des variables financières', fontsize=18)
    plt.tight_layout()
    plt.savefig('figures/correlation_matrix.png', dpi=300)
except Exception as e:
    print(f"Error generating correlation matrix: {e}")
plt.close()

# 4. ANALYSE EN COMPOSANTES PRINCIPALES (PCA)
print("Performing PCA analysis...")
# Sélectionner les colonnes pour la PCA
pca_cols = important_cols[:min(len(important_cols), 15)]  # Limiter à 15 colonnes

# Préparer les données
X = df[pca_cols].copy()
# Gérer les valeurs manquantes
X = X.fillna(X.mean())
# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer la PCA
pca = PCA(n_components=2)  # On se limite à 2 composantes pour la visualisation
X_pca = pca.fit_transform(X_scaled)

# Visualiser les résultats de la PCA
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('Analyse en Composantes Principales', fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/pca_analysis.png', dpi=300)
plt.close()

# Variance expliquée
plt.figure(figsize=(10, 6))
plt.bar(range(1, 3), pca.explained_variance_ratio_)
plt.xlabel('Composante Principale')
plt.ylabel('Variance Expliquée')
plt.title('Variance Expliquée par les Composantes Principales')
plt.xticks([1, 2])
plt.tight_layout()
plt.savefig('figures/pca_explained_variance.png', dpi=300)
plt.close()

# 5. CLUSTERING K-MEANS
print("Performing K-Means clustering...")
# Appliquer K-Means
n_clusters = 3  # On fixe à 3 clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Ajouter les clusters au dataframe
df_with_clusters = df.copy()
df_with_clusters['Cluster'] = clusters

# Visualiser les clusters dans l'espace PCA
plt.figure(figsize=(12, 8))
for i in range(n_clusters):
    plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1], label=f'Cluster {i+1}', alpha=0.7)
plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0], pca.transform(kmeans.cluster_centers_)[:, 1], 
            s=300, c='red', marker='X', label='Centroids')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('Clustering K-Means des entreprises', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/kmeans_clusters.png', dpi=300)
plt.close()

# 6. ANALYSE DES CARACTÉRISTIQUES DES CLUSTERS
print("Analyzing cluster characteristics...")
# Calculer les moyennes des caractéristiques par cluster
if sector_col:
    # Distribution des secteurs par cluster
    sector_by_cluster = pd.crosstab(df_with_clusters['Cluster'], df_with_clusters[sector_col])
    
    # Visualiser cette distribution
    plt.figure(figsize=(15, 10))
    sector_by_cluster.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Distribution des secteurs par cluster', fontsize=16)
    plt.xlabel('Cluster')
    plt.ylabel('Nombre d\'entreprises')
    plt.xticks(rotation=0)
    plt.legend(title='Secteur', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/sector_distribution_by_cluster.png', dpi=300)
    plt.close()

# Moyennes des principales caractéristiques par cluster
plt.figure(figsize=(16, 10))
cluster_means = df_with_clusters.groupby('Cluster')[pca_cols].mean()

# Normalisation pour la visualisation
cluster_means_normalized = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
cluster_means_normalized = cluster_means_normalized.T  # Transpose pour avoir les features en index

sns.heatmap(cluster_means_normalized, annot=False, cmap='YlGnBu', linewidths=0.5)
plt.title('Caractéristiques moyennes par cluster (normalisées)', fontsize=16)
plt.tight_layout()
plt.savefig('figures/cluster_characteristics.png', dpi=300)
plt.close()

print("All visualizations generated successfully and saved to the 'figures' folder!")
