from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from processing import features as ff
from sklearn.linear_model import LogisticRegression
from config.core import config

# for the preprocessors
from sklearn.base import BaseEstimator, TransformerMixin
# for imputation
from feature_engine.imputation import (
    CategoricalImputer,
    AddMissingIndicator,
    MeanMedianImputer)

# for encoding categorical variables
from feature_engine.encoding import (
    RareLabelEncoder,
    OneHotEncoder
)
# set up the pipeline
titanic_pipe = Pipeline([

    # ===== IMPUTATION =====
    # impute categorical variables with string missing
    ('categorical_imputation', CategoricalImputer(
        imputation_method='missing', variables=config['categorical_variables'])),

    # add missing indicator to numerical variables
    ('missing_indicator', AddMissingIndicator(variables=config['numerical_variables'])),

    # impute numerical variables with the median
    ('median_imputation', MeanMedianImputer(
        imputation_method='median', variables=config['numerical_variables'])),


    # Extract letter from cabin
    ('extract_letter', ff.ExtractLetterTransformer(variables=config['cabin'])),


    # == CATEGORICAL ENCODING ======
    # remove categories present in less than 5% of the observations (0.05)
    # group them in one category called 'Rare'
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.05, n_categories=1, variables=config['categorical_variables'])),


    # encode categorical variables using one hot encoding into k-1 variables
    ('categorical_encoder', OneHotEncoder(
        drop_last=True, variables=config['categorical_variables'])),

    # scale
    ('scaler', StandardScaler()),

    ('Logit', LogisticRegression(C=0.0005, random_state=0)),
])
