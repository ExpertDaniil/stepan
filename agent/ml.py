"""Recipe implementations. This module is imported inside training processes."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
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


class NativeHistogramModel(BaseEstimator):
    """Fit-owned categorical codes with an explicit native categorical mask.

    High-cardinality columns use fit-only frequencies, never an arbitrary order.
    Unknown low-cardinality categories become NaN, understood by both backends.
    """
    def __init__(self, regression=False, parameters=None, cpu_threads=1, seed=42, backend='hist'):
        self.regression, self.parameters, self.cpu_threads, self.seed, self.backend = regression, parameters, cpu_threads, seed, backend

    def fit(self, X, y):
        from agent.feature_engineering import _keys
        self.columns_ = list(X.columns)
        self.maps_, self.frequencies_, self.categorical_ = {}, {}, []
        for c in X:
            if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
                self.categorical_.append(False)
                continue
            keys = _keys(X[c])
            unique = sorted(keys.unique())
            if len(unique) <= 254:
                self.maps_[c] = {k: i for i,k in enumerate(unique)}
                self.categorical_.append(True)
            else:
                if len(unique) > 100000:
                    raise ValueError('Categorical column exceeds 100000 fitted keys')
                self.frequencies_[c] = keys.value_counts(normalize=True).to_dict()
                self.categorical_.append(False)
        options = dict(self.parameters or {})
        if self.backend == 'hist':
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            cls = HistGradientBoostingRegressor if self.regression else HistGradientBoostingClassifier
            self.model_ = cls(categorical_features=self.categorical_, random_state=self.seed, **options)
            self.model_.fit(self._frame(X), y)
        elif self.backend == 'lightgbm':
            from lightgbm import LGBMClassifier, LGBMRegressor
            cls = LGBMRegressor if self.regression else LGBMClassifier
            self.model_ = cls(n_jobs=self.cpu_threads, random_state=self.seed, verbosity=-1, **options)
            self.model_.fit(self._frame(X), y, categorical_feature=[i for i,b in enumerate(self.categorical_) if b])
        else:
            raise ValueError('Unsupported native backend')
        if not self.regression:
            self.classes_ = self.model_.classes_
        return self

    def _frame(self, X):
        from agent.feature_engineering import _keys
        if set(X.columns) != set(self.columns_):
            raise ValueError('Native categorical input schema changed')
        result = pd.DataFrame(index=X.index)
        for c in self.columns_:
            if c in self.maps_:
                result[c] = _keys(X[c]).map(self.maps_[c]).astype(float)
            elif c in self.frequencies_:
                result[c] = _keys(X[c]).map(self.frequencies_[c]).fillna(0.).astype(float)
            else:
                result[c] = pd.to_numeric(X[c], errors='coerce').replace([np.inf, -np.inf], np.nan).astype(float)
        return result

    def predict(self, X):
        return self.model_.predict(self._frame(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self._frame(X))


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
    def __init__(self, max_numeric_columns=8, columns=None):
        self.max_numeric_columns = max_numeric_columns
        self.columns = columns

    def fit(self, X, y=None):
        columns = [c for c in X if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
        if self.columns is not None:
            if len(set(self.columns)) != len(self.columns) or not set(self.columns) <= set(columns):
                raise ValueError('NumericInteractions.columns must name existing distinct numeric columns')
            columns = list(self.columns)
            if len(columns) > self.max_numeric_columns:
                raise ValueError('Explicit interactions exceed max_numeric_columns')
        else:
            columns = sorted(columns, key=str)[:self.max_numeric_columns]
        self.selected_columns_ = list(columns)
        self.input_columns_ = list(X.columns)
        prefix = "__automl_interaction_"
        while any(str(c).startswith(prefix) for c in X):
            prefix = "_"+prefix
        self.pairs_ = [(a, b, prefix+str(i)+"_"+str(j)) for i, a in enumerate(columns) for j, b in enumerate(columns[i:], i)]
        self.report_ = {'operator': 'squares_and_pairwise_products', 'inputs': columns,
            'selection_rule': 'explicit_names' if self.columns is not None else 'sorted_numeric_names_then_limit',
            'operations': [{'inputs': [a, b], 'output': name, 'operator': 'product'} for a, b, name in self.pairs_],
            'fit_scope': 'fit_schema', 'missing_or_nonfinite': 'missing', 'units': 'not_inferred'}
        return self

    def transform(self, X):
        if set(X.columns) != set(self.input_columns_):
            raise ValueError('NumericInteractions input schema changed after fit')
        result = X.loc[:, self.input_columns_].copy()
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
    if recipe_id.startswith('tabular-net-'):
        from agent.neural import TabularNetwork
        model=TabularNetwork(regression,params,threads,seed,task_spec.get('_fit_context'),task_spec.get('allowed_labels'))
        return Pipeline([('interactions',NumericInteractions()),('model',model)]) if interactions else model
    if recipe_id.startswith(('hist-native-', 'lightgbm-')):
        model = NativeHistogramModel(regression, params, threads, seed,
            'hist' if recipe_id.startswith('hist-native-') else 'lightgbm')
        return Pipeline([('interactions', NumericInteractions()), ('model', model)]) if interactions else model
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
