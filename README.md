# Projet ML : Prédiction de Survie Multi-Catastrophes Maritimes

Modèle de prédiction de survie combinant les données de trois catastrophes maritimes majeures : le RMS Titanic (1912), le RMS Lusitania (1915) et le MV Sewol (2014).

## Structure du Projet

```
ml-disaster-survival-project/
├── data/
│   ├── raw/                  # Datasets originaux
│   ├── processed/            # Données nettoyées et unifiées
│   └── external/             # Données contextuelles (météo, GDP, specs navires)
├── notebooks/
│   ├── 01_data_collection.ipynb      # Collecte et exploration des données
│   ├── 02_data_processing.ipynb      # Nettoyage et unification
│   ├── 03_feature_engineering.ipynb   # Ingénierie des caractéristiques
│   ├── 04_modeling.ipynb              # Modélisation et comparaison
│   ├── 05_evaluation.ipynb            # Évaluation critique et biais
│   └── 06_technical_report.ipynb      # Rapport technique (export PDF)
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Chargement et nettoyage des données
│   ├── feature_engineering.py # Pipeline de feature engineering
│   ├── models.py              # Modèles et stratégies d'entraînement
│   └── evaluation.py          # Métriques et analyse de biais
├── reports/
│   ├── data_collection_report.md  # Rapport de collecte
│   └── technical_report.pdf       # Rapport technique (généré)
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/<username>/ml-disaster-survival-project.git
cd ml-disaster-survival-project

# Installer les dépendances
pip install -r requirements.txt
```

## Données

Les datasets bruts doivent être placés dans `data/raw/` :
- `Titanic-Dataset.csv` - Dataset Titanic (891 passagers)
- `completemanifest.xls` - Manifeste complet du Lusitania (1961 personnes)
- `sewol_eng.csv` - Données du Sewol traduites en anglais (476 personnes)

Les fichiers de données ne sont pas inclus dans le dépôt Git (voir .gitignore).

## Exécution

Les notebooks doivent être exécutés dans l'ordre suivant :

```bash
cd notebooks/

# 1. Collecte et exploration
jupyter notebook 01_data_collection.ipynb

# 2. Nettoyage et unification
jupyter notebook 02_data_processing.ipynb

# 3. Feature engineering
jupyter notebook 03_feature_engineering.ipynb

# 4. Modélisation
jupyter notebook 04_modeling.ipynb

# 5. Évaluation
jupyter notebook 05_evaluation.ipynb
```

## Méthodologie

1. **Collecte** : 3 catastrophes maritimes (1912, 1915, 2014)
2. **Unification** : Schéma commun avec mapping des variables
3. **Features** : MICE imputation, connectivité sociale, données contextuelles
4. **Modèles** : LR, Random Forest, XGBoost, SVM
5. **Stratégies** : Pooled vs Separate+Ensemble
6. **Évaluation** : Holdout sur le Sewol, analyse de biais, paradoxe éthique

## Résultats Clés

- Le distribution shift entre 1912 et 2014 cause un gap de performance significatif
- Le concept drift (genre→survie) est le facteur le plus impactant
- Le biais "femmes et enfants d'abord" persiste même après suppression directe du genre
- Position éthique : conserver le biais historique avec documentation transparente

## Génération du Rapport PDF

```bash
jupyter nbconvert --to pdf notebooks/06_technical_report.ipynb --output ../reports/technical_report.pdf
```

## Dépendances Principales

- Python 3.10+
- pandas, numpy, scikit-learn
- xgboost, shap
- matplotlib, seaborn
- scipy, statsmodels

Voir `requirements.txt` pour la liste complète.

## Architecture des Modules

```
src/
├── data_processing.py      # Chargement, nettoyage, unification, validation
├── feature_engineering.py   # MICE, connectivité sociale, features contextuelles
├── models.py               # 4 modèles, 2 stratégies, importance, bootstrap CI
└── evaluation.py           # Métriques, biais, concept drift, fairness
```

Chaque module est importable depuis les notebooks via :
```python
sys.path.insert(0, os.path.abspath('..'))
from src.data_processing import load_titanic, clean_titanic
```

## Auteur

Projet réalisé dans le cadre du Master BIDABI - Machine Learning.
