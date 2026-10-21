"""Recipe implementations. This module is imported inside training processes."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def sanitize_frame(frame):
    result = frame.copy()
    for column in result:
        if pd.api.types.is_numeric_dtype(result[column]) and not pd.api.types.is_bool_dtype(result[column]):
            result[column] = pd.to_numeric(result[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            result[column] = result[column].astype("object").where(result[column].notna(), "__MISSING__").astype(str)
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
    if recipe_id.startswith("dummy-"):
        return (DummyRegressor if regression else DummyClassifier)(**params)
    if recipe_id.startswith("catboost-"):
        return CatBoostFrameModel(regression, params, threads, seed)
    numerical = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))])
    if recipe_id.startswith("linear-"):
        numerical.steps.append(("scale", StandardScaler()))
        encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=True)
        estimator = Ridge(solver="lsqr", **params) if regression else LogisticRegression(max_iter=500, random_state=seed, **params)
    elif recipe_id.startswith("hist-gradient-"):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        estimator = (HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier)(random_state=seed, **params)
    elif recipe_id.startswith(("random-forest-", "extra-trees-")):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        family = (ExtraTreesRegressor if regression else ExtraTreesClassifier) if recipe_id.startswith("extra-trees-") else (RandomForestRegressor if regression else RandomForestClassifier)
        estimator = family(n_jobs=threads, random_state=seed, **params)
    else:
        raise ValueError("Unsupported recipe: " + recipe_id)
    categorical = Pipeline([("encode", encoder)])
    prepare = ColumnTransformer([
        ("numeric", numerical, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ], remainder="drop")
    return Pipeline([("sanitize", FunctionTransformer(sanitize_frame, validate=False)),
                     ("prepare", prepare), ("model", estimator)])


def enforce_threads(model, threads):
    if hasattr(model, "get_params") and hasattr(model, "set_params"):
        settings = {k: threads for k in model.get_params(deep=True)
                    if k.split("__")[-1] in {"n_jobs", "thread_count", "num_threads", "cpu_threads"}}
        if settings:
            model.set_params(**settings)
    return model
