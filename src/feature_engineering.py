"""
Module d'ingénierie des caractéristiques pour le projet multi-catastrophes.
Transforme le dataset unifié en features exploitables par les modèles ML.
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"


def load_external_data():
    """Charge les données contextuelles externes."""
    ship_specs = pd.read_csv(EXTERNAL_DATA_DIR / "ship_specifications.csv")
    weather = pd.read_csv(EXTERNAL_DATA_DIR / "weather_conditions.csv")
    economics = pd.read_csv(EXTERNAL_DATA_DIR / "economic_indicators.csv")
    return ship_specs, weather, economics


def add_ship_features(df, ship_specs):
    """
    Ajoute les caractéristiques des navires au dataset.
    Inclut tonnage, ratio canots/passagers, durée du naufrage, etc.
    """
    ship_features = ship_specs[[
        'disaster', 'tonnage_grt', 'total_capacity', 'lifeboat_capacity',
        'sinking_duration_min', 'water_temp_celsius'
    ]].copy()

    # Créer des ratios significatifs
    ship_features['lifeboat_ratio'] = (
        ship_features['lifeboat_capacity'] / ship_features['total_capacity']
    )
    ship_features['sinking_speed'] = (
        ship_features['tonnage_grt'] / ship_features['sinking_duration_min']
    )

    merged = df.merge(ship_features, on='disaster', how='left')
    return merged


def add_weather_features(df, weather):
    """Ajoute les conditions météorologiques."""
    weather_features = weather[[
        'disaster', 'water_temp_celsius', 'wind_speed_knots',
        'wave_height_m', 'daylight'
    ]].copy()
    weather_features = weather_features.rename(columns={
        'water_temp_celsius': 'weather_water_temp'
    })
    weather_features['daylight_binary'] = (
        weather_features['daylight'] == 'day'
    ).astype(int)

    # Éviter le conflit de colonne water_temp_celsius (déjà dans ship_features)
    weather_subset = weather_features[['disaster', 'wind_speed_knots',
                                        'wave_height_m', 'daylight_binary']]
    merged = df.merge(weather_subset, on='disaster', how='left')
    return merged


def add_economic_features(df, economics):
    """Ajoute les indicateurs économiques."""
    # Moyenne par disaster/year (peut y avoir plusieurs pays)
    econ_avg = economics.groupby('year').agg({
        'gdp_per_capita_usd_2014': 'mean',
        'avg_annual_wage_usd_2014': 'mean'
    }).reset_index()

    merged = df.merge(
        econ_avg, left_on='disaster_year', right_on='year', how='left'
    ).drop(columns=['year'], errors='ignore')
    return merged


def create_social_connectivity(df):
    """
    Crée la caractéristique de connectivité sociale.

    Mesure l'avantage de survie lié au voyage en groupe :
    - is_alone : voyage seul
    - group_advantage : score dérivé de la taille du groupe
    - social_network_score : indicateur composite de connexion sociale
    """
    df = df.copy()

    # Déjà présent via create_unified_dataset, mais recalculons proprement
    df['is_alone'] = (df['group_size'] == 1).astype(int)

    # Score d'avantage de groupe (diminishing returns)
    # Les groupes de 2-4 ont le plus d'avantage, au-delà c'est moins bénéfique
    df['group_advantage'] = np.where(
        df['group_size'] == 1, 0,
        np.where(df['group_size'] <= 4, 1,
                 np.where(df['group_size'] <= 8, 0.5, -0.5))
    )

    # Score de connectivité sociale composite
    df['social_network_score'] = (
        (1 - df['is_alone']) * 0.5 +
        df['group_advantage'] * 0.3 +
        (df['is_child'] * (1 - df['is_alone'])) * 0.2  # Enfants en groupe
    )

    return df


def impute_age_mice(df, random_state=42):
    """
    Imputation de l'âge par MICE (Multiple Imputation by Chained Equations).

    MICE est préférable aux indicatrices simples car :
    1. Il utilise les corrélations entre variables pour estimer les valeurs manquantes
    2. Il préserve la distribution conditionnelle de l'âge
    3. Il prend en compte que les données manquantes ne sont pas aléatoires (MNAR)

    Les simples indicatrices de données manquantes sont insuffisantes car :
    - Elles ne récupèrent aucune information sur la valeur probable
    - Elles introduisent un biais si le modèle les utilise comme proxy de classe sociale
    - Elles ne gèrent pas le fait que l'âge manquant est corrélé avec d'autres variables
    """
    df = df.copy()

    # Créer un indicateur AVANT l'imputation (pour analyse MNAR)
    df['age_was_missing'] = df['age'].isna().astype(int)

    # Variables prédictives pour l'imputation
    imputation_vars = ['age', 'sex_binary', 'social_class', 'is_crew',
                        'group_size', 'survived']

    # Extraire le sous-ensemble pour l'imputation
    imputation_df = df[imputation_vars].copy()

    # MICE imputation
    imputer = IterativeImputer(
        max_iter=20,
        random_state=random_state,
        sample_posterior=True
    )
    imputed_values = imputer.fit_transform(imputation_df)

    # Récupérer les âges imputés
    df['age'] = imputed_values[:, 0]

    # Borner les âges à des valeurs réalistes
    df['age'] = df['age'].clip(0, 100)

    return df


def analyze_mnar_patterns(df):
    """
    Analyse les patterns de données manquantes non-aléatoires (MNAR).

    Retourne un dictionnaire avec les résultats de l'analyse.
    """
    results = {}

    if 'age_was_missing' in df.columns:
        # Taux de survie selon que l'âge est manquant ou non
        results['survival_by_age_missing'] = (
            df.groupby('age_was_missing')['survived'].mean().to_dict()
        )

        # Classe sociale selon que l'âge est manquant
        results['class_by_age_missing'] = (
            df.groupby('age_was_missing')['social_class'].mean().to_dict()
        )

        # Par catastrophe
        results['missing_by_disaster'] = (
            df.groupby('disaster')['age_was_missing'].mean().to_dict()
        )

    return results


def create_age_groups(df):
    """Crée des catégories d'âge."""
    df = df.copy()
    bins = [0, 5, 12, 18, 30, 50, 65, 100]
    labels = ['infant', 'child', 'teenager', 'young_adult', 'adult', 'senior', 'elderly']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)
    return df


def create_interaction_features(df):
    """
    Crée des features d'interaction entre variables.
    Ces interactions capturent des effets combinés importants.
    """
    df = df.copy()

    # Interaction sexe × classe (très prédictif pour Titanic)
    df['female_high_class'] = ((df['sex_binary'] == 1) & (df['social_class'] == 1)).astype(int)
    df['male_low_class'] = ((df['sex_binary'] == 0) & (df['social_class'] == 3)).astype(int)

    # Interaction enfant × groupe
    df['child_in_group'] = ((df['is_child'] == 1) & (df['is_alone'] == 0)).astype(int)

    # Interaction crew × catastrophe (via disaster_year comme proxy)
    df['crew_modern'] = ((df['is_crew'] == 1) & (df['disaster_year'] > 1950)).astype(int)

    return df


def prepare_features_for_modeling(df, scale=True):
    """
    Prépare le dataset final pour la modélisation.

    Retourne X (features), y (target) et le scaler utilisé.
    """
    df = df.copy()

    # Features numériques à utiliser
    numeric_features = [
        'age', 'sex_binary', 'social_class', 'is_crew', 'group_size',
        'is_alone', 'is_child', 'group_advantage', 'social_network_score',
        'age_was_missing', 'female_high_class', 'male_low_class',
        'child_in_group', 'crew_modern'
    ]

    # Ajouter les features contextuelles si présentes
    contextual_features = [
        'water_temp_celsius', 'lifeboat_ratio', 'sinking_duration_min',
        'sinking_speed', 'wind_speed_knots', 'wave_height_m',
        'daylight_binary', 'gdp_per_capita_usd_2014'
    ]

    available_features = [f for f in numeric_features + contextual_features
                          if f in df.columns]

    X = df[available_features].copy()
    y = df['survived'].copy()

    # Remplir les NaN restants par la médiane
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled, y, scaler, available_features

    return X, y, scaler, available_features


def full_feature_engineering_pipeline(unified_df):
    """
    Pipeline complète de feature engineering.

    Entrée : dataset unifié (sortie de create_unified_dataset)
    Sortie : dataset enrichi prêt pour la modélisation
    """
    # Charger les données externes
    ship_specs, weather, economics = load_external_data()

    # Ajouter les features contextuelles
    df = add_ship_features(unified_df, ship_specs)
    df = add_weather_features(df, weather)
    df = add_economic_features(df, economics)

    # Connectivité sociale
    df = create_social_connectivity(df)

    # Imputation MICE de l'âge
    df = impute_age_mice(df)

    # Groupes d'âge
    df = create_age_groups(df)

    # Features d'interaction
    df = create_interaction_features(df)

    return df
