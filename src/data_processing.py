"""
Module de traitement et nettoyage des données pour le projet multi-catastrophes.
Fournit des fonctions pour charger, nettoyer et unifier les datasets
Titanic, Lusitania et Sewol.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Chemins par défaut
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"


def load_titanic(filepath=None):
    """Charge le dataset Titanic brut."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "Titanic-Dataset.csv"
    df = pd.read_csv(filepath)
    df['disaster'] = 'titanic'
    df['disaster_year'] = 1912
    return df


def load_lusitania(filepath=None):
    """Charge le dataset Lusitania brut."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "completemanifest.xls"
    df = pd.read_excel(filepath)
    df['disaster'] = 'lusitania'
    df['disaster_year'] = 1915
    return df


def load_sewol(filepath=None):
    """Charge le dataset Sewol brut."""
    if filepath is None:
        filepath = RAW_DATA_DIR / "sewol_eng.csv"
    df = pd.read_csv(filepath)
    df['disaster'] = 'sewol'
    df['disaster_year'] = 2014
    return df


def clean_titanic(df):
    """
    Nettoie le dataset Titanic et le prépare pour l'unification.

    Mapping:
    - Survived : déjà 0/1
    - Pclass : 1=haute, 2=moyenne, 3=basse
    - Sex : déjà male/female
    - group_size : SibSp + Parch + 1
    - is_crew : 0 (tous passagers dans ce dataset)
    """
    cleaned = pd.DataFrame()
    cleaned['survived'] = df['Survived'].astype(int)
    cleaned['age'] = pd.to_numeric(df['Age'], errors='coerce')
    cleaned['sex'] = df['Sex'].str.lower().str.strip()
    cleaned['social_class'] = df['Pclass'].astype(int)
    cleaned['is_crew'] = 0
    cleaned['group_size'] = df['SibSp'] + df['Parch'] + 1
    cleaned['fare'] = df['Fare']
    cleaned['disaster'] = 'titanic'
    cleaned['disaster_year'] = 1912
    cleaned['name'] = df['Name']
    cleaned['cabin'] = df['Cabin']
    cleaned['embarked'] = df['Embarked']
    cleaned['original_index'] = df.index
    return cleaned


def clean_lusitania(df):
    """
    Nettoie le dataset Lusitania et le prépare pour l'unification.

    Mapping:
    - Fate : 'Saved'/'Saved (died from trauma)' → 1, 'Lost' → 0, 'Not on board' → exclure
    - Department/Class → social_class :
        Saloon (1ère classe) → 1
        Second → 2
        Third → 3
        Deck/Engineering/Victualling/Band → crew
    - Passenger/Crew : direct
    - group_size : extrait depuis Traveling Companions
    """
    # Exclure les personnes non à bord
    df = df[df['Fate'] != 'Not on board'].copy()

    cleaned = pd.DataFrame()
    cleaned['survived'] = df['Fate'].map({
        'Saved': 1,
        'Saved (died from trauma)': 0,  # Décédé malgré le sauvetage
        'Lost': 0
    }).astype(int)

    # Nettoyage de l'âge (certaines valeurs contiennent '?')
    cleaned['age'] = df['Age'].apply(_parse_lusitania_age)

    cleaned['sex'] = df['Sex'].str.lower().str.strip()

    # Mapping classe sociale
    class_mapping = {
        'Saloon': 1,      # 1ère classe
        'Second': 2,      # 2ème classe
        'Third': 3,       # 3ème classe
        'Third (Distressed British Seaman)': 3,
        'Stowaway': 3,    # Clandestins → basse classe
    }
    crew_departments = ['Deck', 'Engineering', 'Victualling', 'Band']

    cleaned['social_class'] = df['Department/Class'].map(class_mapping)
    cleaned['is_crew'] = df['Passenger/Crew'].apply(
        lambda x: 1 if x == 'Crew' else 0
    )
    # Pour l'équipage, assigner classe 2 par défaut (moyenne)
    cleaned.loc[cleaned['is_crew'] == 1, 'social_class'] = 2

    cleaned['fare'] = np.nan  # Non disponible pour Lusitania

    # Taille du groupe depuis les notes de voyage
    cleaned['group_size'] = df['Traveling Companions and other notes'].apply(
        _extract_lusitania_group_size
    )

    cleaned['disaster'] = 'lusitania'
    cleaned['disaster_year'] = 1915
    cleaned['name'] = df['Family name'] + ', ' + df['Title'] + ' ' + df['Personal name']
    cleaned['citizenship'] = df['Citizenship']
    cleaned['lifeboat'] = df['Lifeboat']
    cleaned['adult_minor'] = df['Adult/Minor']
    cleaned['original_index'] = df.index
    return cleaned


def _parse_lusitania_age(age_value):
    """Parse les valeurs d'âge du Lusitania qui peuvent contenir des '?'."""
    if pd.isna(age_value):
        return np.nan
    age_str = str(age_value).strip()
    # Retirer les '?' et espaces
    age_str = age_str.replace('?', '').replace(' ', '')
    if age_str == '' or age_str == 'Infant':
        return np.nan
    try:
        return float(age_str)
    except ValueError:
        return np.nan


def _extract_lusitania_group_size(notes):
    """
    Estime la taille du groupe depuis les notes de voyage du Lusitania.
    Si des compagnons sont mentionnés, compte les virgules + 1 comme proxy.
    """
    if pd.isna(notes):
        return 1  # Voyage seul par défaut
    notes = str(notes)
    if notes.strip() == '':
        return 1
    # Compter les mentions de personnes (approximation par les virgules)
    companions = notes.count(',') + 1
    return min(companions + 1, 10)  # +1 pour la personne elle-même, cap à 10


def clean_sewol(df):
    """
    Nettoie le dataset Sewol et le prépare pour l'unification.

    Mapping:
    - Raw : 'survival' → 1, 'Dead' → 0
    - Category-1 : sailor → crew, student/Normal → passenger
    - social_class : sailor → 2, student → 3, Normal (passagers réguliers) → selon travel type
    - group_size : depuis Category-3 (Family, Group, personal)
    """
    cleaned = pd.DataFrame()
    cleaned['survived'] = df['Raw'].map({
        'survival': 1,
        'Dead': 0
    }).astype(int)

    cleaned['age'] = pd.to_numeric(df['age'], errors='coerce')
    cleaned['sex'] = df['gender'].str.lower().str.strip()

    # Mapping classe sociale pour le Sewol
    # sailor (équipage) → 2, student (lycéens en voyage scolaire) → 3, Normal → selon type
    social_class_map = {'sailor': 2, 'student': 3, 'Normal': 2}
    cleaned['social_class'] = df['Category-1'].map(social_class_map).fillna(2).astype(int)

    cleaned['is_crew'] = (df['Category-1'] == 'sailor').astype(int)
    cleaned['fare'] = np.nan  # Non disponible

    # Taille du groupe depuis Category-3
    cleaned['group_size'] = df['Category-3'].apply(_extract_sewol_group_size)

    # Caractéristiques spécifiques au Sewol
    cleaned['floor'] = pd.to_numeric(df['floor'], errors='coerce')
    cleaned['location'] = df['location']

    cleaned['disaster'] = 'sewol'
    cleaned['disaster_year'] = 2014
    cleaned['name'] = np.nan  # Pas de noms dans le dataset
    cleaned['original_index'] = df.index
    return cleaned


def _extract_sewol_group_size(category3):
    """Estime la taille du groupe depuis Category-3 du Sewol."""
    if pd.isna(category3):
        return 1
    cat = str(category3).lower()
    if 'family' in cat:
        return 3  # Estimation famille
    if 'group' in cat or 'colleagues' in cat or 'travel' in cat:
        return 4  # Estimation groupe
    if 'half' in cat:
        return 30  # Classe d'école (demi-classe)
    if 'personal' in cat:
        return 1
    return 1


def create_unified_dataset(titanic_clean, lusitania_clean, sewol_clean):
    """
    Crée un dataset unifié avec les colonnes communes.

    Colonnes retenues : survived, age, sex, social_class, is_crew,
    group_size, disaster, disaster_year
    """
    common_cols = [
        'survived', 'age', 'sex', 'social_class', 'is_crew',
        'group_size', 'disaster', 'disaster_year'
    ]

    unified = pd.concat([
        titanic_clean[common_cols],
        lusitania_clean[common_cols],
        sewol_clean[common_cols]
    ], ignore_index=True)

    # Créer des features dérivées
    unified['is_alone'] = (unified['group_size'] == 1).astype(int)
    unified['is_child'] = (unified['age'] < 18).astype(int)
    unified['sex_binary'] = (unified['sex'] == 'female').astype(int)

    return unified


def save_processed_data(df, filename, output_dir=None):
    """Sauvegarde un DataFrame nettoyé dans le dossier processed."""
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    print(f"Sauvegardé : {filepath} ({len(df)} lignes)")
    return filepath


def get_data_quality_report(df, name="Dataset"):
    """Génère un rapport de qualité des données."""
    report = {
        'name': name,
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'duplicates': df.duplicated().sum()
    }
    return report
