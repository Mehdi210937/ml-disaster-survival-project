"""
Module d'évaluation critique pour le projet multi-catastrophes.
Inclut le diagnostic du gap de performance, l'analyse de biais,
et les métriques avancées.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve


def comprehensive_evaluation(model, X_test, y_test, model_name="Model"):
    """
    Évaluation complète d'un modèle avec toutes les métriques pertinentes.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

    if y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_test, y_proba)
        results['avg_precision'] = average_precision_score(y_test, y_proba)
        results['brier_score'] = brier_score_loss(y_test, y_proba)

        # Courbes
        results['roc_curve'] = roc_curve(y_test, y_proba)
        results['pr_curve'] = precision_recall_curve(y_test, y_proba)
        results['calibration'] = calibration_curve(y_test, y_proba, n_bins=10)

    return results


def compute_disparate_impact(y_pred, protected_attribute, reference_group=None):
    """
    Calcule le ratio d'impact disparate pour une variable protégée.

    Le ratio d'impact disparate est défini comme :
        P(prediction positive | groupe défavorisé) / P(prediction positive | groupe favorisé)

    Un ratio < 0.8 est généralement considéré comme discriminatoire
    (règle des 4/5 de l'EEOC).
    """
    groups = np.unique(protected_attribute)

    if reference_group is None:
        # Le groupe de référence est celui avec le taux de prédiction positive le plus élevé
        group_rates = {}
        for g in groups:
            mask = protected_attribute == g
            if mask.sum() > 0:
                group_rates[g] = y_pred[mask].mean()
        reference_group = max(group_rates, key=group_rates.get)

    ref_rate = y_pred[protected_attribute == reference_group].mean()

    disparate_impact = {}
    for g in groups:
        mask = protected_attribute == g
        if mask.sum() > 0:
            group_rate = y_pred[mask].mean()
            di_ratio = group_rate / ref_rate if ref_rate > 0 else np.nan
            disparate_impact[g] = {
                'positive_rate': group_rate,
                'count': mask.sum(),
                'disparate_impact_ratio': di_ratio,
                'is_discriminatory': di_ratio < 0.8 if not np.isnan(di_ratio) else None
            }

    return disparate_impact, reference_group


def analyze_bias_comprehensive(model, X_test, y_test, df_test):
    """
    Analyse complète des biais selon genre, classe sociale et âge.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    bias_results = {}

    # 1. Biais de genre
    if 'sex' in df_test.columns:
        gender = df_test['sex'].values
        di_gender, ref = compute_disparate_impact(y_pred, gender)
        bias_results['gender'] = {
            'disparate_impact': di_gender,
            'reference_group': ref,
            'actual_survival': {
                g: y_test[gender == g].mean() for g in np.unique(gender)
            }
        }

    # 2. Biais de classe sociale
    if 'social_class' in df_test.columns:
        social_class = df_test['social_class'].values
        di_class, ref = compute_disparate_impact(y_pred, social_class)
        bias_results['social_class'] = {
            'disparate_impact': di_class,
            'reference_group': ref,
            'actual_survival': {
                g: y_test[social_class == g].mean() for g in np.unique(social_class)
            }
        }

    # 3. Biais d'âge
    if 'age' in df_test.columns:
        age_groups = pd.cut(
            df_test['age'].values, bins=[0, 18, 30, 50, 100],
            labels=['0-17', '18-29', '30-49', '50+']
        ).astype(str)
        di_age, ref = compute_disparate_impact(y_pred, age_groups)
        bias_results['age_group'] = {
            'disparate_impact': di_age,
            'reference_group': ref,
            'actual_survival': {
                g: y_test.values[age_groups == g].mean() for g in np.unique(age_groups)
            }
        }

    return bias_results


def diagnose_performance_gap(model, X_train, y_train, X_test, y_test,
                              df_train, df_test, feature_names):
    """
    Diagnostic du gap de performance entre CV et test holdout.

    Identifie si la baisse est due à :
    1. Distributions de features différentes
    2. Frontières de décision différentes
    3. Features manquantes
    4. Taille d'échantillon
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    # CV score
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')

    # Test score
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)

    diagnostic = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score,
        'gap': cv_scores.mean() - test_score,
    }

    # 1. Distribution shift
    feature_shifts = {}
    for feat in feature_names:
        if feat in X_train.columns and feat in X_test.columns:
            from scipy.stats import ks_2samp
            stat, p_val = ks_2samp(
                X_train[feat].dropna(),
                X_test[feat].dropna()
            )
            feature_shifts[feat] = {'ks_statistic': stat, 'p_value': p_val}
    diagnostic['feature_distribution_shifts'] = feature_shifts

    # 2. Base rates
    diagnostic['train_base_rate'] = y_train.mean()
    diagnostic['test_base_rate'] = y_test.mean()

    # 3. Confidence calibration
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        diagnostic['mean_confidence'] = y_proba.mean()
        diagnostic['confidence_std'] = y_proba.std()

    return diagnostic


def covariate_shift_analysis(X_train, X_test, feature_names):
    """
    Analyse détaillée du covariate shift entre train et test.
    Utilise le test de Kolmogorov-Smirnov pour chaque feature.
    """
    from scipy.stats import ks_2samp

    results = []
    for feat in feature_names:
        stat, p_val = ks_2samp(X_train[feat].values, X_test[feat].values)
        results.append({
            'feature': feat,
            'ks_statistic': stat,
            'p_value': p_val,
            'significant_shift': p_val < 0.05,
            'train_mean': X_train[feat].mean(),
            'test_mean': X_test[feat].mean(),
            'train_std': X_train[feat].std(),
            'test_std': X_test[feat].std()
        })

    return pd.DataFrame(results).sort_values('ks_statistic', ascending=False)


def concept_drift_analysis(df_train, df_test):
    """
    Analyse de la dérive conceptuelle entre les données d'entraînement et de test.
    Compare les relations conditionnelles features→survie entre les deux ensembles.
    """
    results = {}

    # Relation genre → survie
    if 'sex' in df_train.columns and 'survived' in df_train.columns:
        train_gender = df_train.groupby('sex')['survived'].mean()
        test_gender = df_test.groupby('sex')['survived'].mean()
        results['gender_survival'] = {
            'train': train_gender.to_dict(),
            'test': test_gender.to_dict(),
            'drift': {sex: abs(train_gender.get(sex, 0) - test_gender.get(sex, 0))
                      for sex in set(train_gender.index) | set(test_gender.index)}
        }

    # Relation classe → survie
    if 'social_class' in df_train.columns:
        train_class = df_train.groupby('social_class')['survived'].mean()
        test_class = df_test.groupby('social_class')['survived'].mean()
        results['class_survival'] = {
            'train': train_class.to_dict(),
            'test': test_class.to_dict(),
            'drift': {sc: abs(train_class.get(sc, 0) - test_class.get(sc, 0))
                      for sc in set(train_class.index) | set(test_class.index)}
        }

    # Relation équipage → survie
    if 'is_crew' in df_train.columns:
        train_crew = df_train.groupby('is_crew')['survived'].mean()
        test_crew = df_test.groupby('is_crew')['survived'].mean()
        results['crew_survival'] = {
            'train': train_crew.to_dict(),
            'test': test_crew.to_dict(),
            'drift': {c: abs(train_crew.get(c, 0) - test_crew.get(c, 0))
                      for c in set(train_crew.index) | set(test_crew.index)}
        }

    return results


def fairness_metrics(y_true, y_pred, protected_attribute):
    """
    Calcule les métriques d'équité standard.

    Retourne:
    - Statistical parity difference
    - Equal opportunity difference
    - Predictive parity difference
    """
    groups = np.unique(protected_attribute)
    if len(groups) < 2:
        return {}

    group_metrics = {}
    for g in groups:
        mask = protected_attribute == g
        if mask.sum() == 0:
            continue
        group_metrics[g] = {
            'positive_rate': y_pred[mask].mean(),
            'tpr': recall_score(y_true[mask], y_pred[mask], zero_division=0),
            'ppv': precision_score(y_true[mask], y_pred[mask], zero_division=0)
        }

    if len(group_metrics) < 2:
        return {}

    groups_list = list(group_metrics.keys())
    ref = groups_list[0]

    results = {
        'statistical_parity_diff': abs(
            group_metrics[groups_list[0]]['positive_rate'] -
            group_metrics[groups_list[1]]['positive_rate']
        ),
        'equal_opportunity_diff': abs(
            group_metrics[groups_list[0]]['tpr'] -
            group_metrics[groups_list[1]]['tpr']
        ),
        'predictive_parity_diff': abs(
            group_metrics[groups_list[0]]['ppv'] -
            group_metrics[groups_list[1]]['ppv']
        ),
        'group_metrics': group_metrics
    }

    return results
