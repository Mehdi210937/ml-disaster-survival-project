# Rapport de Collecte de Données

## 1. Sources des Données

### 1.1 RMS Titanic (1912)
- **Source** : Kaggle Titanic Dataset (original des archives de la British Board of Trade)
- **URL** : https://www.kaggle.com/datasets/yasserh/titanic-dataset
- **Date d'accès** : Février 2025
- **Format** : CSV (Titanic-Dataset.csv)
- **Taille** : 891 lignes x 12 colonnes
- **Fiabilité** : Haute - données largement vérifiées et utilisées dans la communauté ML depuis des décennies. Sous-ensemble du manifeste complet (~2224 personnes).

### 1.2 RMS Lusitania (1915)
- **Source** : Manifeste complet du RMS Lusitania (compilé à partir d'archives historiques)
- **Référence** : RMS Lusitania Resource, Encyclopedia Titanica, National Archives UK
- **Date d'accès** : Février 2025
- **Format** : XLS (completemanifest.xls)
- **Taille** : 1961 lignes x 23 colonnes
- **Fiabilité** : Haute - compilation exhaustive incluant passagers et équipage. Certaines valeurs d'âge comportent des '?' indiquant une incertitude historique.

### 1.3 MV Sewol (2014)
- **Source** : Données compilées depuis les rapports du Korean Maritime Safety Tribunal et les médias coréens
- **Référence** : Board of Audit and Inspection of Korea, rapports d'enquête gouvernementaux
- **Date d'accès** : Février 2025
- **Format** : CSV (sewol_eng.csv) - traduit du coréen vers l'anglais
- **Taille** : 476 lignes x 9 colonnes
- **Fiabilité** : Moyenne - traduction automatique visible dans certaines catégories. Données vérifiables via les rapports officiels coréens.

### 1.4 Données Contextuelles Externes
- **Spécifications des navires** : Jane's Fighting Ships, registres maritimes historiques
- **Conditions météorologiques** : Archives NOAA, rapports d'enquête des catastrophes
- **Indicateurs économiques** : Banque mondiale, données historiques du PIB

## 2. Évaluation de la Qualité des Données

### 2.1 Titanic
| Aspect | Évaluation |
|--------|-----------|
| Complétude | 80% (Age manquant 19.9%, Cabin 77.1%, Embarked 0.2%) |
| Exactitude | Haute (données vérifiées historiquement) |
| Cohérence | Bonne (format standardisé, peu d'anomalies) |
| Temporalité | Données statiques de 1912 |

**Note sur les données manquantes** : L'absence de la cabine est fortement corrélée avec la 3ème classe (MNAR - Missing Not At Random), ce qui reflète le fait que les passagers de 3ème classe n'avaient pas de cabines assignées dans les registres.

### 2.2 Lusitania
| Aspect | Évaluation |
|--------|-----------|
| Complétude | 65% (beaucoup de champs optionnels vides : City, County, Lifeboat) |
| Exactitude | Moyenne-Haute (quelques âges incertains marqués '?') |
| Cohérence | Bonne (23 colonnes bien structurées) |
| Temporalité | Données statiques de 1915 |

**Note** : Le Lusitania est le dataset le plus complet en termes de personnes (1960 à bord) et de colonnes (citoyenneté, position, canot de sauvetage).

### 2.3 Sewol
| Aspect | Évaluation |
|--------|-----------|
| Complétude | 90% (peu de valeurs manquantes, surtout sur l'âge) |
| Exactitude | Moyenne (artefacts de traduction anglaise, catégories parfois ambiguës) |
| Cohérence | Moyenne (noms de catégories traduits de manière incohérente) |
| Temporalité | Données statiques de 2014 |

**Note** : La forte proportion d'étudiants (325/476 = 68%) crée un déséquilibre démographique unique par rapport aux autres catastrophes.

## 3. Processus de Nettoyage

### 3.1 Transformations appliquées

**Variable cible** :
- Titanic : `Survived` (déjà 0/1)
- Lusitania : `Fate` → `survived` (Saved=1, Lost=0, Saved(died from trauma)=0, Not on board=exclu)
- Sewol : `Raw` → `survived` (survival=1, Dead=0)

**Âge** :
- Titanic : Conservation des valeurs numériques, NaN pour les manquants
- Lusitania : Suppression des '?' et conversion en float
- Sewol : Conversion directe en float

**Classe sociale (unifiée 1-3)** :
- Titanic : Pclass directement
- Lusitania : Saloon→1, Second→2, Third→3, Équipage→2
- Sewol : Équipage→2, Étudiants→3, Passagers normaux→2

### 3.2 Justification des choix
- L'équipage est assigné à la classe 2 car il a un accès intermédiaire aux ressources du navire
- Les étudiants du Sewol sont en classe 3 car ils n'avaient pas d'autonomie dans l'évacuation
- Les passagers normaux du Sewol sont en classe 2 (ferry domestique = classe moyenne)

## 4. Considérations Éthiques

### 4.1 Respect des victimes
Les données utilisées concernent des personnes réelles ayant vécu des événements tragiques. Cette analyse est menée dans un cadre strictement académique avec le respect approprié pour les victimes et leurs familles.

### 4.2 Données publiques
- Les manifestes du Titanic et du Lusitania sont des documents historiques publics, disponibles dans les archives nationales.
- Les données du Sewol proviennent de rapports d'enquête gouvernementaux rendus publics.

### 4.3 Utilisation responsable
- Les noms des individus ne sont pas utilisés comme features prédictives.
- L'analyse vise à comprendre les facteurs de survie pour améliorer la sécurité maritime, pas pour juger les victimes.
- Les biais historiques (discrimination de classe, de genre) sont documentés et discutés de manière critique.

### 4.4 Limites éthiques
- Le modèle prédictif ne doit pas être utilisé pour des décisions de triage en situation réelle sans validation approfondie.
- Les corrélations trouvées reflètent des contextes historiques spécifiques et ne doivent pas être généralisées sans précaution.

## 5. Limitations des Données

1. **Biais de sélection** : Le dataset Titanic ne contient que 891 des ~2224 personnes à bord.
2. **Biais de traduction** : Le dataset Sewol a été traduit du coréen, ce qui introduit des ambiguïtés.
3. **Hétérogénéité des features** : Les trois datasets n'ont pas les mêmes variables disponibles.
4. **Données manquantes non aléatoires** : L'absence de certaines informations est liée au statut social (MNAR).
5. **Granularité temporelle** : Pas d'information sur le moment précis du sauvetage de chaque individu.
