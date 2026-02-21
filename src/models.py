"""
Module de modélisation pour le projet multi-catastrophes.
Implémente les 4 modèles de base et les 2 stratégies d'entraînement.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, classification_report,
    confusion_matrix
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')


def get_base_models(random_state=42):
    """
    Retourne les 4 modèles de base.

    1. Régression Logistique : baseline interprétable, coefficients lisibles
    2. Random Forest : ensemble robuste, gère bien les non-linéarités
    3. XGBoost : gradient boosting SOTA pour données tabulaires
    4. SVM : frontières de décision flexibles avec kernel RBF
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            C=1.0,
            solver='lbfgs'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=random_state
        )
    }
    return models


def evaluate_model(model, X_test, y_test):
    """Évalue un modèle sur un jeu de test avec plusieurs métriques."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        metrics['brier_score'] = brier_score_loss(y_test, y_proba)

    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    return metrics


def cross_validate_models(models, X, y, cv=5, random_state=42):
    """
    Évalue les modèles par validation croisée stratifiée.
    Retourne un DataFrame avec les scores pour chaque modèle.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = {}

    for name, model in models.items():
        scores = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            cv_scores = cross_val_score(
                model, X, y, cv=skf, scoring=metric, n_jobs=-1
            )
            scores[f'{metric}_mean'] = cv_scores.mean()
            scores[f'{metric}_std'] = cv_scores.std()
        results[name] = scores

    return pd.DataFrame(results).T


def compute_learning_curves(models, X, y, cv=5, random_state=42):
    """
    Calcule les courbes d'apprentissage pour chaque modèle.
    Montre comment la performance évolue avec la quantité de données.
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    curves = {}

    for name, model in models.items():
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1,
            random_state=random_state
        )
        curves[name] = {
            'train_sizes': train_sizes_abs,
            'train_mean': train_scores.mean(axis=1),
            'train_std': train_scores.std(axis=1),
            'val_mean': val_scores.mean(axis=1),
            'val_std': val_scores.std(axis=1)
        }

    return curves


# ========== Stratégie 1 : Pooled ==========

def train_pooled_strategy(models, X_train, y_train, X_test, y_test):
    """
    Stratégie regroupée : entraîner chaque modèle sur toutes les catastrophes.

    Avantages :
    - Plus de données d'entraînement
    - Le modèle peut apprendre des patterns universels
    Inconvénients :
    - Suppose une relation features→survie identique entre catastrophes
    - Peut être biaisé vers la catastrophe la plus représentée
    """
    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model

    return results, trained_models


# ========== Stratégie 2 : Separate + Ensemble ==========

def train_separate_strategy(base_model_fn, X_train, y_train, disaster_train,
                            X_test, y_test, disaster_test):
    """
    Stratégie séparée : entraîner un modèle par catastrophe, puis combiner.

    Avantages :
    - Chaque modèle est spécialisé pour sa catastrophe
    - Pas de conflit entre patterns contradictoires
    Inconvénients :
    - Moins de données par modèle
    - Pas de transfert de connaissance entre catastrophes
    """
    disaster_models = {}
    disasters = disaster_train.unique()

    for disaster in disasters:
        mask = disaster_train == disaster
        X_d = X_train[mask]
        y_d = y_train[mask]

        if len(y_d.unique()) < 2:
            continue

        model = base_model_fn()
        model.fit(X_d, y_d)
        disaster_models[disaster] = model

    # Prédiction : utiliser le modèle le plus proche ou la moyenne
    y_pred_all = []
    y_proba_all = []

    for disaster in disasters:
        if disaster in disaster_models:
            model = disaster_models[disaster]
            y_pred_all.append(model.predict(X_test))
            if hasattr(model, 'predict_proba'):
                y_proba_all.append(model.predict_proba(X_test)[:, 1])

    # Ensemble par vote majoritaire
    if y_pred_all:
        y_pred_ensemble = np.round(np.mean(y_pred_all, axis=0)).astype(int)
        y_proba_ensemble = np.mean(y_proba_all, axis=0) if y_proba_all else None
    else:
        y_pred_ensemble = np.zeros(len(y_test))
        y_proba_ensemble = None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_ensemble),
        'precision': precision_score(y_test, y_pred_ensemble, zero_division=0),
        'recall': recall_score(y_test, y_pred_ensemble, zero_division=0),
        'f1': f1_score(y_test, y_pred_ensemble, zero_division=0),
    }
    if y_proba_ensemble is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba_ensemble)
        metrics['brier_score'] = brier_score_loss(y_test, y_proba_ensemble)
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_ensemble)

    return metrics, disaster_models


# ========== Importance des caractéristiques ==========

def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """Importance par permutation."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats,
        random_state=random_state, n_jobs=-1
    )
    return pd.Series(
        result.importances_mean, index=X.columns
    ).sort_values(ascending=False)


def compute_drop_column_importance(model_fn, X, y, cv=5, random_state=42):
    """
    Importance par suppression de colonne.
    Entraîne le modèle sans chaque feature et mesure la dégradation.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Score baseline avec toutes les features
    baseline_model = model_fn()
    baseline_scores = cross_val_score(baseline_model, X, y, cv=skf, scoring='accuracy')
    baseline = baseline_scores.mean()

    importances = {}
    for col in X.columns:
        X_dropped = X.drop(columns=[col])
        dropped_model = model_fn()
        dropped_scores = cross_val_score(dropped_model, X_dropped, y, cv=skf, scoring='accuracy')
        importances[col] = baseline - dropped_scores.mean()

    return pd.Series(importances).sort_values(ascending=False)


def get_logistic_coefficients(model, feature_names):
    """Retourne les coefficients de la régression logistique."""
    if hasattr(model, 'coef_'):
        return pd.Series(
            model.coef_[0], index=feature_names
        ).sort_values(key=abs, ascending=False)
    return pd.Series()


def compare_feature_importance_methods(model, model_fn, X, y, feature_names):
    """
    Compare les 4 méthodes d'importance des features.
    Retourne un DataFrame avec les résultats de chaque méthode.
    """
    results = pd.DataFrame(index=feature_names)

    # 1. Permutation importance
    perm_imp = compute_permutation_importance(model, X, y)
    results['permutation'] = perm_imp

    # 2. Coefficients (si logistic regression)
    coefs = get_logistic_coefficients(model, feature_names)
    if len(coefs) > 0:
        results['coefficients'] = coefs.abs()

    # 3. Drop-column importance
    drop_imp = compute_drop_column_importance(model_fn, X, y)
    results['drop_column'] = drop_imp

    # 4. Tree-based importance (si disponible)
    if hasattr(model, 'feature_importances_'):
        results['tree_importance'] = pd.Series(
            model.feature_importances_, index=feature_names
        )

    return results


def mcnemar_test(y_true, y_pred_1, y_pred_2):
    """
    Test de McNemar pour comparer deux classifieurs.
    Retourne la statistique de test et la p-value.
    """
    from scipy.stats import chi2

    # Construire la table de contingence
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)

    b = np.sum(correct_1 & ~correct_2)  # 1 correct, 2 incorrect
    c = np.sum(~correct_1 & correct_2)  # 1 incorrect, 2 correct

    if b + c == 0:
        return 0.0, 1.0

    # Statistique avec correction de continuité
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return statistic, p_value


def bootstrap_confidence_interval(model, X_test, y_test, metric_fn, n_bootstrap=1000,
                                   confidence=0.95, random_state=42):
    """
    Calcule un intervalle de confiance par bootstrap pour une métrique donnée.
    Utile pour quantifier l'incertitude des métriques de performance.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_test)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        X_boot = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
        y_boot = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        y_pred_boot = model.predict(X_boot)
        scores.append(metric_fn(y_boot, y_pred_boot))

    alpha = 1 - confidence
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'lower': lower,
        'upper': upper,
        'confidence': confidence
    }


def generate_model_comparison_table(pooled_results, separate_results):
    """
    Génère un tableau de comparaison propre entre les deux stratégies.
    """
    rows = []
    for model_name in pooled_results:
        row = {'Model': model_name}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in pooled_results[model_name]:
                row[f'Pooled_{metric}'] = pooled_results[model_name][metric]
            if metric in separate_results.get(model_name, {}):
                row[f'Separate_{metric}'] = separate_results[model_name][metric]
        rows.append(row)

    return pd.DataFrame(rows).set_index('Model')
