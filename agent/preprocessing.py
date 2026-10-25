"""Owned preprocessing policy shared by construction and researcher context."""

MISSING_CATEGORY = '__MISSING__'


def preprocessing_policy(recipe_id):
    if recipe_id.startswith('python-pipeline-'):
        return {'numeric_missing': 'defined_by_source', 'categorical_encoding': 'defined_by_source',
                'numeric_scaling': False, 'missing_indicators': 'defined_by_source'}
    if recipe_id.startswith('dummy-'):
        return {'numeric_missing': 'unused', 'categorical_encoding': 'unused',
                'numeric_scaling': False, 'missing_indicators': False}
    if recipe_id.startswith('catboost-'):
        return {'numeric_missing': 'native_nan', 'categorical_encoding': 'native_string_categories',
                'categorical_missing_value': MISSING_CATEGORY,
                'numeric_scaling': False, 'missing_indicators': False}
    if recipe_id.startswith(('hist-native-', 'lightgbm-')):
        return {'numeric_missing': 'native_nan', 'categorical_encoding': 'native_mask_fit_only_codes',
                'high_cardinality': 'fit_frequency_above_254_categories', 'unknown_category': 'native_nan',
                'numeric_scaling': False, 'missing_indicators': False}
    return {'numeric_missing': 'median',
            'categorical_encoding': 'one_hot' if recipe_id.startswith('linear-') else 'ordinal',
            'categorical_missing_value': MISSING_CATEGORY,
            'numeric_scaling': recipe_id.startswith('linear-'), 'missing_indicators': False}
