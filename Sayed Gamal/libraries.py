from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer


import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import Image, display

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC


from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, f_classif
# Define custom transformers


class OptionalScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scale=True):
        self.scale = scale
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        if self.scale:
            self.scaler.fit(X, y)
        return self

    def transform(self, X):
        if self.scale:
            return self.scaler.transform(X)
        return X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_column].replace({'2018-2-29': '3/1/2018'}, inplace=True)
        X[self.date_column] = pd.to_datetime(
            X[self.date_column], errors='coerce', format='%m/%d/%Y')
        X['reservation_year'] = X[self.date_column].dt.year
        X['reservation_month'] = X[self.date_column].dt.month
        X['reservation_day'] = X[self.date_column].dt.day
        X.drop(columns=[self.date_column], inplace=True)
        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            lower = (X[col] >= lower_bound)
            upper = (X[col] <= upper_bound)
            X.loc[~upper, col] = upper_bound
            X.loc[~lower, col] = lower_bound
        return X


def multiply_by_factor(X, factor=0.01):
    X = X.copy()
    X[['P-C', 'P-not-C']] = X[['P-C', 'P-not-C']] * factor
    return X


def transform_multiply_by_factor(X):
    return multiply_by_factor(X)
