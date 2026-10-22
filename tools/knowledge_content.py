"""Project-authored transferable guidance and unexecuted complete pipeline examples.

These strings are examples for the researcher, never automatically imported or run.
Upstream write-ups are separately preserved and explicitly labelled unverified.
"""

MISSING_EXAMPLE = '''from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from agent.ml import build_pipeline

class MissingFlags(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if X[c].isna().any()]
        self.prefix_ = "missing_flag_"
        while any(str(c).startswith(self.prefix_) for c in X.columns):
            self.prefix_ = "_" + self.prefix_
        return self

    def transform(self, X):
        out = X.copy()
        for i, column in enumerate(self.columns_):
            out[self.prefix_ + str(i)] = X[column].isna().astype(float)
        return out

def make_pipeline(task_spec, params):
    options = dict(params)
    default = "hist-gradient-regressor-v1" if task_spec["task_kind"] == "regression" else "hist-gradient-classifier-v1"
    base = options.pop("base_recipe_id", default)
    return Pipeline([("missing_flags", MissingFlags()),
                     ("model", build_pipeline(base, task_spec, options))])
'''

FREQUENCY_EXAMPLE = '''from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from agent.ml import build_pipeline

class FrequencyFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.maps_ = {}
        for column in X.select_dtypes(exclude="number").columns:
            self.maps_[column] = X[column].astype(str).value_counts(normalize=True).to_dict()
        self.prefix_ = "frequency_"
        while any(str(c).startswith(self.prefix_) for c in X.columns):
            self.prefix_ = "_" + self.prefix_
        return self

    def transform(self, X):
        out = X.copy()
        for i, column in enumerate(self.maps_):
            out[self.prefix_ + str(i)] = X[column].astype(str).map(self.maps_[column]).fillna(0.0)
        return out

def make_pipeline(task_spec, params):
    options = dict(params)
    default = "hist-gradient-regressor-v1" if task_spec["task_kind"] == "regression" else "hist-gradient-classifier-v1"
    base = options.pop("base_recipe_id", default)
    return Pipeline([("frequency", FrequencyFeatures()),
                     ("model", build_pipeline(base, task_spec, options))])
'''

LOG_EXAMPLE = '''import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from agent.ml import build_pipeline

class LogColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = list(self.columns)
        for column in self.columns_:
            if (X[column].dropna() < 0).any():
                raise ValueError("log1p example requires nonnegative selected features")
        return self

    def transform(self, X):
        out = X.copy()
        for column in self.columns_:
            if (X[column].dropna() < 0).any():
                raise ValueError("Negative unseen values violate the selected log1p precondition")
            out[column] = np.log1p(X[column])
        return out

def make_pipeline(task_spec, params):
    options = dict(params)
    columns = options.pop("log_columns", [])
    default = "linear-regressor-v1" if task_spec["task_kind"] == "regression" else "linear-classifier-v1"
    base = options.pop("base_recipe_id", default)
    return Pipeline([("log_features", LogColumns(columns)),
                     ("model", build_pipeline(base, task_spec, options))])
'''

CONSTANT_EXAMPLE = '''from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from agent.ml import build_pipeline

class DropConstants(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = [c for c in X.columns if X[c].nunique(dropna=False) > 1]
        if not self.columns_:
            raise ValueError("Every feature is constant on fit rows")
        return self

    def transform(self, X):
        return X.loc[:, self.columns_].copy()

def make_pipeline(task_spec, params):
    options = dict(params)
    default = "hist-gradient-regressor-v1" if task_spec["task_kind"] == "regression" else "hist-gradient-classifier-v1"
    base = options.pop("base_recipe_id", default)
    return Pipeline([("drop_constants", DropConstants()),
                     ("model", build_pipeline(base, task_spec, options))])
'''


def method_cards():
    definitions = [
        ('constant-features', 'Remove features constant on fit rows', ['constant_features'],
         'Many constant or all-missing features; no target values are required.',
         'Fit the retained column set on fit rows, then use exactly that set for validation/test.',
         'Do not decide which columns to retain using held-out labels or test variation. This is a complexity reduction, not a promised score gain.', CONSTANT_EXAMPLE),
        ('missing-flags', 'Preserve missingness before imputation', ['missing'],
         'Missing features and an owned median-imputation base without indicators.',
         'Add row-wise missing flags before the fitted imputer. Select columns using fit rows only.',
         'Native-NaN CatBoost already preserves numeric missingness; median-imputation is not its limitation. Missingness can shift between train and test.', MISSING_EXAMPLE),
        ('frequency-features', 'Fit-only frequency features for categories', ['frequency_encoding', 'categorical', 'rare_categories'],
         'Repeated categorical levels where prevalence may convey information.',
         'Fit relative-frequency maps on fit rows, add frequencies, map unknown levels to zero, retain original features.',
         'No target values are used. Frequency can reflect sampling drift; do not learn maps from validation or test.', FREQUENCY_EXAMPLE),
        ('log-numeric-features', 'Logarithms for selected nonnegative features', ['log_features', 'linear'],
         'Nonnegative counts or skewed magnitudes; inspect distributions before selecting columns.',
         'Apply log1p to explicitly selected columns, then fit preprocessing and a suitable linear model.',
         'This transforms features, not the target. Do not transform arbitrary signed measurements or assume an isolated gain from a blended Kaggle solution.', LOG_EXAMPLE),
        ('native-categories', 'Native categories versus ordinal prefixes', ['categorical', 'boosting'],
         'Categories with no meaningful order; compare native category models or fitted one-hot representations.',
         'Owned CatBoost uses string categories and numeric NaN directly. HGB/tree recipes ordinal-encode categories; linear recipes one-hot encode.',
         'Never describe all catalogue recipes as median/ordinal. One-hot may be expensive at high cardinality.', None),
        ('rare-levels', 'Rare and unseen category handling', ['rare_categories', 'categorical'],
         'Many low-count levels or unseen levels in test features.',
         'Fit category vocabularies and optional rare-level grouping on fit rows, preserve an explicit unknown path.',
         'Do not build a train+validation vocabulary. Do not import the source competition\'s thresholds unchanged.', None),
        ('target-transform', 'Choose target transformations under the actual metric', ['target_transform'],
         'Regression with skewed targets or scale-dependent errors.',
         'Fit the target transformation on fit labels only and invert predictions before authoritative evaluation.',
         'RMSE, RMSLE and MAE optimize different errors. Log-target training can bias original-scale predictions. Use current measured comparisons.', None),
        ('interactions', 'Candidate interactions from data structure', ['interactions'],
         'A plausible relationship among features, even if marginal correlations look weak.',
         'Propose bounded products, ratios with safe denominators, or domain-motivated combinations jointly with an appropriate model.',
         'Low pairwise association does not establish independence or absence of useful interactions. Prevent feature explosion and nonfinite values.', None),
        ('group-aggregates', 'Learned aggregates respecting split boundaries', ['aggregates'],
         'Repeated entities, groups or sessions with usable group identifiers.',
         'Fit feature-only group statistics on fit rows and define an unknown-group fallback.',
         'Target aggregates require out-of-fold construction inside fit. Entity leakage can make a random holdout misleading.', None),
        ('temporal-features', 'Temporal features with an inference-time boundary', ['temporal', 'validation'],
         'Timestamps or repeated measurements with a real temporal prediction setting.',
         'Use calendar features and only lags/rolling statistics available at prediction time.',
         'The mere presence of a year column does not prove a forecasting task. Future observations and labels cannot enter features.', None),
        ('split-structure', 'Match validation to repeated entities and time', ['validation', 'leakage'],
         'Entity reuse, duplicates or chronological drift.',
         'Inspect whether the frozen validation reflects deployment. Explain a mismatch explicitly before proposing actions.',
         'Researchers cannot mutate the frozen protocol. A Kaggle random-fold recipe is not authority for the current split.', None),
        ('class-imbalance', 'Multiclass imbalance and decision rules', ['imbalance'],
         'Multiclass labels with unequal representation or poor minority-class recall.',
         'Inspect class support and confusion patterns; consider regularization, class weights or training-only sampling under the requested metric.',
         'Accuracy, macro-F1 and log-loss can disagree. Class weighting changes probability calibration; binary thresholds do not transfer directly.', None),
        ('error-slices', 'Use error slices to select a next experiment', ['error_analysis'],
         'A measured model with errors concentrated in interpretable groups.',
         'Connect an error pattern to a feature/model hypothesis and measure the complete proposed solution.',
         'Group bias alone does not prove underfitting. Repeated use of validation diagnostics can overfit the holdout.', None),
        ('linear-baseline', 'Encoded and regularized linear alternatives', ['linear', 'categorical'],
         'Sparse categories, additive structure or a cheap complementary model.',
         'Fit one-hot categories, scale numeric columns inside fit and tune regularization on the frozen protocol.',
         'Binary AUC scores from category-encoding competitions are not multiclass performance evidence.', None),
        ('boosting-budget', 'Budget-aware boosting capacity search', ['boosting'],
         'A measured boosting base whose capacity or regularization remains uncertain.',
         'Coordinate learning rate, iterations, leaf capacity and regularization using measured durations and scores.',
         'Iteration counts are not comparable across CatBoost and HGB. A tuning series is optional; preserve time for refit.', None),
        ('complementary-ensemble', 'Blend genuinely complementary predictions', ['ensemble', 'validation'],
         'Several measured, compatible models with complementary errors and remaining compute.',
         'Use predictions aligned to the same protocol; train stacking layers only on out-of-fold base predictions.',
         'A 33-model, multi-layer winning ensemble is not a three-hour recipe. In-sample stacking leaks labels. No gain is assumed.', None),
        ('leakage-check', 'Keep supervised feature learning inside fit', ['leakage', 'validation'],
         'Target encoding, feature selection, target transforms or supervised preprocessing.',
         'Fit every learned operation within the returned pipeline. Build training encodings out of fold when they use targets.',
         'Never use held-out labels or source-competition leaks. A reported leaderboard improvement does not certify honest validation.', None),
    ]
    cards = [{'id': 'method-'+key, 'kind': 'method', 'title': title, 'tags': tags,
             'task_kinds': ['regression'] if key == 'target-transform' else ['multiclass_classification'] if key == 'class-imbalance' else ['regression', 'multiclass_classification'],
             'when': when, 'mechanism': mechanism, 'cautions': cautions,
             'origin': 'project_authored_guidance', 'status': 'hypothesis_not_current_task_evidence',
             'code_example': {'origin': 'project_authored', 'verification': 'not_executed', 'source': code} if code else None}
            for key, title, tags, when, mechanism, cautions, code in definitions]
    cards[0]['inspected_code_reference'] = {
        'source_record': 348, 'solution_title': '11th place',
        'url': 'https://github.com/Danila89/kaggle_mercedes/blob/30d9a6e95977b68d23971aba2919fc2a82259161/solution_11_place.ipynb',
        'git_blob_sha1': '32c50e6f517842165c8057d4d6923007b43d9a19',
        'code_cells': 7, 'status': 'inspected_legacy_notebook; project_example_is_an_independent_adaptation',
        'adaptation_notes': 'The notebook removes constant features, but also contains target-dependent grouping, train/test distribution filtering, local helper imports, file IO, Python-2 syntax and 100 repeated model fits. Those parts are not copied into the runtime example. Supervised grouping would need fit-only/OOF handling. Original notebook is linked, not bundled or advertised as runnable Python 3.12.'}
    return cards
