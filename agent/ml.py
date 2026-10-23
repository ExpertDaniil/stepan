"""Recipe implementations. This module is imported inside training processes."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from agent.preprocessing import MISSING_CATEGORY, preprocessing_policy


def sanitize_frame(frame):
    result = frame.copy()
    for column in result:
        if pd.api.types.is_numeric_dtype(result[column]) and not pd.api.types.is_bool_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            result[column] = result[column].astype("object").where(result[column].notna(), MISSING_CATEGORY).astype(str)
    return result


class CatBoostFrameModel(BaseEstimator):
    def __init__(self, regression=False, parameters=None, cpu_threads=1, seed=42):
        self.regression, self.parameters, self.cpu_threads, self.seed = regression, parameters, cpu_threads, seed

    def fit(self, X, y):
        from catboost import CatBoostClassifier, CatBoostRegressor
        frame = sanitize_frame(X)
        categorical = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])]
        model_type = CatBoostRegressor if self.regression else CatBoostClassifier
        self.model_ = model_type(**dict(self.parameters or {}), thread_count=self.cpu_threads,
                                 random_seed=self.seed, verbose=False, allow_writing_files=False)
        self.model_.fit(frame, y, cat_features=categorical)
        if not self.regression:
            self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(sanitize_frame(X))).reshape(-1)

    def predict_proba(self, X):
        if self.regression:
            raise TypeError('Regression does not produce class probabilities')
        return self.model_.predict_proba(sanitize_frame(X))


class SequentialTargets(BaseEstimator):
    """Independent targets with one active estimator fit at a time per lease."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        from copy import deepcopy
        values = np.asarray(y, dtype=float)
        if values.ndim != 2 or values.shape[1] < 2:
            raise ValueError('SequentialTargets requires multiple numeric targets')
        self.models_ = []
        for index in range(values.shape[1]):
            model = deepcopy(self.estimator)
            model.fit(X, values[:, index])
            self.models_.append(model)
        return self

    def predict(self, X):
        return np.column_stack([np.asarray(model.predict(X)).reshape(-1) for model in self.models_])


class EncodedClassifier(ClassifierMixin, BaseEstimator):
    """Adapt XGBoost's consecutive integer classes to the task's original labels."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.label_encoder_.classes_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, self.label_encoder_.transform(y))
        return self

    def predict(self, X):
        indices = np.asarray(self.estimator_.predict(X), dtype=int)
        return self.label_encoder_.inverse_transform(indices)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class ShiftedLogTargetModel(BaseEstimator):
    """Target transform fitted only on fit rows; predictions return original units."""
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all():
            raise ValueError("Target transform requires finite labels")
        self.shift_ = max(0.0, -float(y.min()))
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.log1p(y + self.shift_))
        return self

    def predict(self, X):
        return np.expm1(self.estimator_.predict(X)) - self.shift_


class NumericInteractions(TransformerMixin, BaseEstimator):
    """Bounded squares/products. Column selection is fitted only on training rows."""
    def __init__(self, max_numeric_columns=8):
        self.max_numeric_columns = max_numeric_columns

    def fit(self, X, y=None):
        columns = [c for c in X if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
        columns = columns[:self.max_numeric_columns]
        prefix = "__automl_interaction_"
        while any(str(c).startswith(prefix) for c in X):
            prefix = "_"+prefix
        self.pairs_ = [(a, b, prefix+str(i)+"_"+str(j)) for i, a in enumerate(columns) for j, b in enumerate(columns[i:], i)]
        return self

    def transform(self, X):
        result = X.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            for a, b, name in self.pairs_:
                values = pd.to_numeric(X[a], errors="coerce") * pd.to_numeric(X[b], errors="coerce")
                result[name] = values.replace([np.inf, -np.inf], np.nan)
        return result


def validate_estimator(model):
    """Validate sklearn parameter constraints before entering the fit stage."""
    objects = [model]
    if hasattr(model, "get_params"):
        objects.extend(model.get_params(deep=True).values())
    seen = set()
    for item in objects:
        if id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, "_parameter_constraints") and callable(getattr(item, "_validate_params", None)):
            item._validate_params()
    return model


def build_pipeline(recipe_id, task_spec, params):
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
    params = dict(params)
    threads = int(params.pop("_cpu_threads", 1))
    seed = int(params.pop("_seed", 42))
    regression = task_spec["task_kind"] == "regression"
    transform = params.pop("target_transform", "identity")
    if transform not in {"identity", "shifted_log1p"} or (not regression and transform != "identity"):
        raise ValueError("Unsupported target transform")
    if transform == "shifted_log1p":
        return ShiftedLogTargetModel(build_pipeline(recipe_id, task_spec, {**params, "_cpu_threads": threads, "_seed": seed}))
    interactions = params.pop("numeric_interactions", False)
    if not isinstance(interactions, bool):
        raise ValueError("numeric_interactions must be boolean")
    if recipe_id.startswith("dummy-"):
        return (DummyRegressor if regression else DummyClassifier)(**params)
    if recipe_id.startswith("catboost-"):
        model = CatBoostFrameModel(regression, params, threads, seed)
        return Pipeline([("interactions", NumericInteractions()), ("model", model)]) if interactions else model
    policy = preprocessing_policy(recipe_id)
    numerical = Pipeline([("impute", SimpleImputer(strategy=policy['numeric_missing'], keep_empty_features=True))])
    encoder = (OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
               if policy['categorical_encoding'] == 'one_hot'
               else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    if recipe_id.startswith("linear-"):
        estimator = Ridge(solver="lsqr", **params) if regression else LogisticRegression(max_iter=500, random_state=seed, **params)
    elif recipe_id.startswith("hist-gradient-"):
        estimator = (HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier)(random_state=seed, **params)
    elif recipe_id.startswith(("random-forest-", "extra-trees-")):
        family = (ExtraTreesRegressor if regression else ExtraTreesClassifier) if recipe_id.startswith("extra-trees-") else (RandomForestRegressor if regression else RandomForestClassifier)
        estimator = family(n_jobs=threads, random_state=seed, **params)
    elif recipe_id.startswith('lightgbm-'):
        from lightgbm import LGBMClassifier, LGBMRegressor
        family = LGBMRegressor if regression else LGBMClassifier
        estimator = family(n_jobs=threads, random_state=seed, device_type='cpu',
                           deterministic=True, force_col_wise=True, verbosity=-1, **params)
    elif recipe_id.startswith('xgboost-'):
        from xgboost import XGBClassifier, XGBRegressor
        family = XGBRegressor if regression else XGBClassifier
        estimator = family(n_jobs=threads, random_state=seed, device='cpu', tree_method='hist',
                           verbosity=0, **params)
        if not regression:
            estimator = EncodedClassifier(estimator)
    else:
        raise ValueError("Unsupported recipe: " + recipe_id)
    if policy['numeric_scaling']:
        numerical.steps.append(("scale", StandardScaler()))
    categorical = Pipeline([("encode", encoder)])
    prepare = ColumnTransformer([
        ("numeric", numerical, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ], remainder="drop")
    steps = [("sanitize", FunctionTransformer(sanitize_frame, validate=False))]
    if interactions:
        steps.append(("interactions", NumericInteractions()))
    return Pipeline(steps + [("prepare", prepare), ("model", estimator)])


def enforce_threads(model, threads):
    if hasattr(model, "get_params") and hasattr(model, "set_params"):
        settings = {k: threads for k in model.get_params(deep=True)
                    if k.split("__")[-1] in {"n_jobs", "thread_count", "num_threads", "cpu_threads"}}
        if settings:
            model.set_params(**settings)
    return model
