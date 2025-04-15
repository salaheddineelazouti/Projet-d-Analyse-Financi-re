import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
os.makedirs('figures', exist_ok=True)
print("Generating improved visualizations for financial analysis report...")

# Chargement des données
try:
    # Essayer d'abord avec les données contenant les ratios
    df = pd.read_csv('financial_data_with_ratios.csv')
    print(f"Loaded financial_data_with_ratios.csv with {df.shape[0]} rows and {df.shape[1]} columns")
except:
    try:
        # Sinon avec les données nettoyées
        df = pd.read_csv('cleaned_financial_data.csv')
        print(f"Loaded cleaned_financial_data.csv with {df.shape[0]} rows and {df.shape[1]} columns")
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
print(f"Found {len(numeric_cols)} numeric columns")

# Fonction améliorée pour nettoyer les valeurs aberrantes
def clean_outliers(df, columns, method='percentile', low=0.01, high=0.99):
    """
    Nettoie les valeurs aberrantes dans un DataFrame
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame à nettoyer
    columns : list
        Liste des colonnes à nettoyer
    method : str
        Méthode de nettoyage ('percentile', 'zscore' ou 'iqr')
    low : float
        Percentile bas (pour la méthode percentile) ou facteur multiplicatif (pour IQR)
    high : float
        Percentile haut (pour la méthode percentile) ou facteur multiplicatif (pour IQR)
        
    Returns:
    --------
    pandas DataFrame
        DataFrame nettoyé
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        # Ignorer les colonnes non numériques
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Ignorer les colonnes avec trop de valeurs manquantes
        if df[col].isna().sum() > 0.5 * len(df):
            continue
            
        # Remplacer les valeurs infinies par NaN
        df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
        
        # Méthode basée sur les percentiles
        if method == 'percentile':
            lower_bound = df_clean[col].quantile(low)
            upper_bound = df_clean[col].quantile(high)
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Méthode basée sur le z-score
        elif method == 'zscore':
            mean_val = df_clean[col].mean()
            std_val = df_clean[col].std()
            if std_val > 0:  # Éviter division par zéro
                z_scores = np.abs((df_clean[col] - mean_val) / std_val)
                df_clean.loc[z_scores > 3, col] = np.nan  # Remplacer par NaN les z-scores > 3
                
        # Méthode basée sur l'IQR (écart interquartile)
        elif method == 'iqr':
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (low * iqr)
            upper_bound = q3 + (high * iqr)
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

# Nettoyer les valeurs aberrantes
print("Cleaning outliers from numeric columns...")
df_clean = clean_outliers(df, numeric_cols, method='percentile', low=0.01, high=0.99)

# 1. VISUALISATION DES ENTREPRISES PAR SECTEUR
print("Generating sector distribution visualization...")
plt.figure(figsize=(14, 10))
if sector_col in df_clean.columns:
    sector_counts = df_clean[sector_col].value_counts()
    # Limiter le nombre de secteurs affichés si trop nombreux
    if len(sector_counts) > 15:
        sector_counts = sector_counts.head(15)
        title_suffix = " (top 15)"
    else:
        title_suffix = ""
        
    ax = sns.barplot(y=sector_counts.index, x=sector_counts.values)
    plt.title(f'Nombre d\'entreprises par secteur{title_suffix}', fontsize=18)
    plt.xlabel('Nombre d\'entreprises', fontsize=14)
    plt.ylabel('Secteur', fontsize=14)
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(sector_counts.values):
        ax.text(v + 0.5, i, str(v), va='center')
        
    plt.tight_layout()
    plt.savefig('figures/companies_by_sector.png', dpi=300)
    plt.close()

# 2. ANALYSE DES RATIOS FINANCIERS
print("Analyzing financial ratios...")
# Chercher des colonnes qui pourraient être des ratios financiers
ratio_keywords = ['ROA', 'ROE', 'margin', 'ratio', 'turnover']
ratio_cols = []

for keyword in ratio_keywords:
    for col in numeric_cols:
        if keyword.lower() in col.lower() and col not in ratio_cols:
            # Vérifier que les valeurs sont dans une plage raisonnable pour un ratio
            if abs(df_clean[col].median()) < 1000:  # Filtre basique pour éliminer les colonnes non-ratio
                ratio_cols.append(col)

# Limiter à 5 ratios les plus pertinents
if len(ratio_cols) > 5:
    # Préférence pour ROA, ROE et les marges
    priority_ratios = ['ROA', 'ROE', 'Net_Margin', 'Gross_Margin', 'Operating_Margin']
    selected_ratios = []
    
    # D'abord ajouter les ratios prioritaires s'ils existent
    for priority in priority_ratios:
        for ratio in ratio_cols:
            if priority.lower() in ratio.lower() and ratio not in selected_ratios:
                selected_ratios.append(ratio)
                if len(selected_ratios) >= 5:
                    break
        if len(selected_ratios) >= 5:
            break
    
    # Compléter avec d'autres ratios si nécessaire
    if len(selected_ratios) < 5:
        for ratio in ratio_cols:
            if ratio not in selected_ratios:
                selected_ratios.append(ratio)
                if len(selected_ratios) >= 5:
                    break
    
    ratio_cols = selected_ratios

# Si on a trouvé des ratios, on les analyse
if ratio_cols:
    print(f"Found {len(ratio_cols)} ratio columns: {ratio_cols}")
    
    # Visualisation des ratios par secteur
    for ratio in ratio_cols[:3]:  # On limite à 3 ratios pour ne pas surcharger
        plt.figure(figsize=(14, 10))
        try:
            # Filtrer les valeurs aberrantes spécifiquement pour ce graphique
            temp_df = df_clean.copy()
            temp_df[ratio] = temp_df[ratio].clip(
                lower=temp_df[ratio].quantile(0.01),
                upper=temp_df[ratio].quantile(0.99)
            )
            
            # Boxplot avec échelle limitée
            ax = sns.boxplot(x=sector_col, y=ratio, data=temp_df)
            plt.title(f'{ratio} par secteur', fontsize=18)
            plt.xticks(rotation=90)
            
            # Ajuster l'échelle de l'axe y pour qu'elle soit raisonnable
            y_min, y_max = plt.ylim()
            if y_max > 100:  # Si l'échelle est trop grande
                plt.ylim(y_min, min(y_max, temp_df[ratio].quantile(0.95) * 1.2))
            
            plt.tight_layout()
            plt.savefig(f'figures/{ratio}_by_sector.png', dpi=300)
            plt.close()
            
            # Histogramme de distribution du ratio
            plt.figure(figsize=(12, 8))
            sns.histplot(temp_df[ratio].dropna(), kde=True)
            plt.title(f'Distribution de {ratio}', fontsize=18)
            plt.xlabel(ratio, fontsize=14)
            plt.ylabel('Fréquence', fontsize=14)
            plt.tight_layout()
            plt.savefig(f'figures/{ratio}_distribution.png', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error generating visualization for {ratio}: {e}")
            plt.close()
else:
    print("No financial ratio columns found")

# 3. ANALYSE DES CORRÉLATIONS
print("Generating correlation matrix...")
# Sélectionner un sous-ensemble de colonnes numériques pour l'analyse de corrélation
if len(numeric_cols) > 15:
    # Donner la priorité aux ratios financiers et métriques importantes
    important_cols = ratio_cols.copy() if ratio_cols else []
    
    # Ajouter d'autres colonnes importantes
    keywords = ['revenue', 'profit', 'income', 'assets', 'equity', 'liabilities', 'cash', 'market']
    for keyword in keywords:
        for col in numeric_cols:
            if keyword.lower() in col.lower() and col not in important_cols:
                important_cols.append(col)
                if len(important_cols) >= 15:
                    break
        if len(important_cols) >= 15:
            break
    
    # Si on n'a pas assez de colonnes, on complète avec d'autres colonnes numériques
    if len(important_cols) < 15:
        for col in numeric_cols:
            if col not in important_cols:
                important_cols.append(col)
                if len(important_cols) >= 15:
                    break
    
    corr_cols = important_cols
else:
    corr_cols = numeric_cols

# Créer la matrice de corrélation
plt.figure(figsize=(16, 14))
try:
    # Remplacer les valeurs manquantes pour le calcul de la corrélation
    corr_df = df_clean[corr_cols].copy()
    corr_df = corr_df.fillna(corr_df.mean())
    
    # Calculer la matrice de corrélation
    corr_matrix = corr_df.corr()
    
    # Créer un masque pour le triangle supérieur
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Créer la heatmap avec le masque
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt='.2f', 
        linewidths=0.5,
        mask=mask,
        square=True,
        vmin=-1, 
        vmax=1
    )
    plt.title('Matrice de corrélation des variables financières', fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figures/correlation_matrix.png', dpi=300)
except Exception as e:
    print(f"Error generating correlation matrix: {e}")
plt.close()

# 4. ANALYSE EN COMPOSANTES PRINCIPALES (PCA)
print("Performing PCA analysis...")
# Sélectionner les colonnes pour la PCA
pca_cols = corr_cols.copy()

# Préparer les données
X = df_clean[pca_cols].copy()

# Gérer les valeurs manquantes
X = X.fillna(X.mean())

# Normaliser les données avec RobustScaler qui est moins sensible aux outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer la PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualiser les résultats de la PCA
plt.figure(figsize=(12, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolor='k', s=40)
plt.grid(True)
plt.xlabel('Première composante principale', fontsize=14)
plt.ylabel('Deuxième composante principale', fontsize=14)
plt.title('Analyse en Composantes Principales', fontsize=18)

# Ajouter des annotations pour quelques points si possible
if 'Ticker' in df_clean.columns:
    # Sélectionner quelques points éloignés pour les annoter
    distances = np.sum(X_pca**2, axis=1)
    furthest_points = np.argsort(distances)[-5:]  # 5 points les plus éloignés
    
    for idx in furthest_points:
        plt.annotate(
            df_clean['Ticker'].iloc[idx],
            (X_pca[idx, 0], X_pca[idx, 1]),
            xytext=(5, 5),
            textcoords='offset points'
        )

plt.tight_layout()
plt.savefig('figures/pca_analysis.png', dpi=300)
plt.close()

# Variance expliquée
plt.figure(figsize=(10, 6))
plt.bar(range(1, 3), pca.explained_variance_ratio_)
plt.xlabel('Composante Principale', fontsize=14)
plt.ylabel('Variance Expliquée', fontsize=14)
plt.title('Variance Expliquée par les Composantes Principales', fontsize=18)
plt.xticks([1, 2])
plt.grid(True, axis='y')
# Ajouter les pourcentages en annotations
for i, v in enumerate(pca.explained_variance_ratio_):
    plt.text(i+1, v+0.01, f'{v:.2%}', ha='center')
plt.tight_layout()
plt.savefig('figures/pca_explained_variance.png', dpi=300)
plt.close()

# Loadings de PCA (importance des variables)
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=pca_cols
)

# Trier les loadings par importance absolue pour PC1
pc1_loadings = loadings['PC1'].abs().sort_values(ascending=False)
top_features_pc1 = pc1_loadings.head(10).index

# Visualiser les loadings des variables les plus importantes pour PC1
plt.figure(figsize=(12, 8))
loadings.loc[top_features_pc1, 'PC1'].sort_values().plot(kind='barh')
plt.title('Contribution des variables à la première composante principale', fontsize=18)
plt.xlabel('Coefficient', fontsize=14)
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('figures/pca_loadings_pc1.png', dpi=300)
plt.close()

# 5. CLUSTERING K-MEANS
print("Performing K-Means clustering...")
# Appliquer K-Means
n_clusters = 3  # On fixe à 3 clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Ajouter les clusters au dataframe
df_with_clusters = df_clean.copy()
df_with_clusters['Cluster'] = clusters

# Visualiser les clusters dans l'espace PCA
plt.figure(figsize=(12, 10))
colors = ['#4285F4', '#EA4335', '#FBBC05']  # Bleu, Rouge, Jaune (couleurs Google)
markers = ['o', 's', '^']  # Cercle, Carré, Triangle

# Tracer chaque cluster avec sa propre couleur et forme
for i in range(n_clusters):
    plt.scatter(
        X_pca[clusters == i, 0], 
        X_pca[clusters == i, 1], 
        c=colors[i], 
        marker=markers[i],
        s=80,
        alpha=0.7,
        edgecolor='k',
        linewidth=0.5,
        label=f'Cluster {i+1}'
    )

# Ajouter les centres des clusters transformés dans l'espace PCA
centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centers_pca[:, 0], 
    centers_pca[:, 1], 
    s=200, 
    c='red', 
    marker='X', 
    label='Centroïdes',
    edgecolor='k',
    linewidth=1.5
)

plt.xlabel('Première composante principale', fontsize=14)
plt.ylabel('Deuxième composante principale', fontsize=14)
plt.title('Clustering K-Means des entreprises', fontsize=18)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('figures/kmeans_clusters.png', dpi=300)
plt.close()

# 6. ANALYSE DES CARACTÉRISTIQUES DES CLUSTERS
print("Analyzing cluster characteristics...")
# Calculer les moyennes des caractéristiques par cluster
if sector_col in df_with_clusters.columns:
    # Distribution des secteurs par cluster
    sector_by_cluster = pd.crosstab(df_with_clusters['Cluster'], df_with_clusters[sector_col])
    sector_by_cluster_pct = sector_by_cluster.div(sector_by_cluster.sum(axis=1), axis=0)
    
    # Visualiser cette distribution
    plt.figure(figsize=(16, 10))
    sector_by_cluster.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='tab20')
    plt.title('Distribution des secteurs par cluster', fontsize=18)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Nombre d\'entreprises', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Secteur', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/sector_distribution_by_cluster.png', dpi=300)
    plt.close()
    
    # Visualiser la distribution en pourcentage
    plt.figure(figsize=(16, 10))
    sector_by_cluster_pct.plot(kind='bar', stacked=True, ax=plt.gca(), cmap='tab20')
    plt.title('Composition sectorielle des clusters (en %)', fontsize=18)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Pourcentage', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Secteur', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('figures/sector_percentage_by_cluster.png', dpi=300)
    plt.close()

# Moyennes des principales caractéristiques par cluster
if ratio_cols:
    # Sélectionner des ratios financiers clés pour la comparaison
    key_ratios = ratio_cols[:min(len(ratio_cols), 10)]
    
    # Calculer la moyenne de chaque ratio par cluster
    cluster_means = df_with_clusters.groupby('Cluster')[key_ratios].mean()
    
    # Visualiser les ratios par cluster
    plt.figure(figsize=(14, 10))
    cluster_means.plot(kind='bar', ax=plt.gca())
    plt.title('Ratios financiers moyens par cluster', fontsize=18)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Valeur moyenne', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Ratio', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('figures/financial_ratios_by_cluster.png', dpi=300)
    plt.close()
    
    # Visualisation en radar chart pour comparer les clusters
    from math import pi
    
    # Normaliser les données pour le radar chart
    cluster_means_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())
    
    # Créer la figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Nombre de variables
    N = len(key_ratios)
    
    # Angle de chaque axe
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Fermer le polygone
    
    # Ajouter les labels
    plt.xticks(angles[:-1], key_ratios, size=12)
    
    # Tracer pour chaque cluster
    for i in range(len(cluster_means_norm)):
        values = cluster_means_norm.iloc[i].tolist()
        values += values[:1]  # Fermer le polygone
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {i}')
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Profil financier des clusters (normalisé)', fontsize=18)
    plt.tight_layout()
    plt.savefig('figures/cluster_radar_chart.png', dpi=300)
    plt.close()

print("All improved visualizations generated successfully and saved to the 'figures' folder!")
