import os
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter
import sys

# Définir le répertoire du projet
project_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(project_dir, "resultats")
notebooks_dir = os.path.join(project_dir, "notebooks")

# Créer le répertoire des résultats s'il n'existe pas déjà
os.makedirs(results_dir, exist_ok=True)

# Liste des notebooks à exécuter
notebooks = [
    "1_data_loading_exploration.ipynb",
    "2_financial_ratios_visualization.ipynb",
    "3_clustering_and_pca.ipynb",
    "4_predictive_modeling.ipynb",
    "5_conclusions_and_interpretations.ipynb"
]

# Fonction pour exécuter un notebook et enregistrer les résultats
def execute_notebook(notebook_path, output_path):
    print(f"Exécution du notebook: {notebook_path}")
    
    try:
        # Charger le notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Configurer l'exécuteur
        ep = ExecutePreprocessor(timeout=600, kernel_name='finance_analysis')
        
        # Exécuter le notebook
        ep.preprocess(nb, {'metadata': {'path': project_dir}})
        
        # Enregistrer le notebook exécuté
        executed_notebook_path = os.path.join(results_dir, os.path.basename(notebook_path))
        with open(executed_notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        # Convertir le notebook en HTML
        html_exporter = HTMLExporter()
        (body, resources) = html_exporter.from_notebook_node(nb)
        
        # Enregistrer le HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body)
        
        print(f"Notebook exécuté et résultats enregistrés dans: {output_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de l'exécution du notebook {notebook_path}: {e}")
        return False

# Fonction principale
def main():
    print("Démarrage de l'exécution des notebooks...")
    
    # Copier les fichiers du dataset dans le répertoire du projet si nécessaire
    # (Cette étape est prise en charge dans le premier notebook)
    
    # Exécuter chaque notebook séquentiellement
    for notebook in notebooks:
        notebook_path = os.path.join(notebooks_dir, notebook)
        output_path = os.path.join(results_dir, notebook.replace('.ipynb', '.html'))
        
        success = execute_notebook(notebook_path, output_path)
        if not success:
            print(f"Arrêt de l'exécution en raison d'une erreur dans le notebook: {notebook}")
            break
    
    print("Création d'un index des résultats...")
    create_index_html()
    
    print("Terminé!")

# Créer une page d'index HTML pour les résultats
def create_index_html():
    index_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Résultats de l'Analyse Financière</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; }
        .notebook-link { 
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
        a { text-decoration: none; color: #3498db; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Résultats de l'Analyse Financière</h1>
    <p>Voici les résultats de l'exécution des notebooks pour l'analyse financière avec Data Science et Machine Learning.</p>
    
    <h2>Notebooks</h2>
"""
    
    # Ajouter des liens vers chaque notebook
    for notebook in notebooks:
        html_file = notebook.replace('.ipynb', '.html')
        index_content += f'    <div class="notebook-link"><a href="{html_file}">{notebook}</a></div>\n'
    
    index_content += """
    <h2>Fichiers CSV générés</h2>
"""
    
    # Ajouter des liens vers les fichiers CSV générés
    csv_files = [f for f in os.listdir(project_dir) if f.endswith('.csv')]
    for csv_file in csv_files:
        # Copier le fichier CSV dans le répertoire des résultats
        source_path = os.path.join(project_dir, csv_file)
        target_path = os.path.join(results_dir, csv_file)
        try:
            with open(source_path, 'r') as src:
                with open(target_path, 'w') as dst:
                    dst.write(src.read())
            index_content += f'    <div class="notebook-link"><a href="{csv_file}">{csv_file}</a></div>\n'
        except Exception as e:
            print(f"Erreur lors de la copie du fichier CSV {csv_file}: {e}")
    
    index_content += """
</body>
</html>
"""
    
    with open(os.path.join(results_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_content)

if __name__ == "__main__":
    main()
