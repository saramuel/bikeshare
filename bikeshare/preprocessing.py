# coding: utf-8
from datetime import timedelta

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from statsmodels.tsa.seasonal import seasonal_decompose


class LogTransform(BaseEstimator, TransformerMixin):
    """
    Reduces influence of outliers on regression by log-transforming the target.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log1p(X)

    def inverse_transform(self, X):
        return np.expm1(X)


# NOTE: not deriving from BaseEstimator here, since we want GridSearchCV to
# ignore this class because we fit it manually.
class TrendRemover(TransformerMixin):
    def __init__(self, remove_trend=True):
        """
        Compute the yearly time trend and perform standardization by dividing the target
        by the time trend (if remove_trend is True).

        The transform and fit member functions operate on a pandas Series with TimeSeriesIndex.

        Parameters:
        ----------
        remove_trend : if False, don't do anything
        """
        self.remove_trend = remove_trend

    def fit(self, X, y=None):
        """Compute the time trend to be used for later scaling.
        Parameters
        ----------
        X : Series with TimeSeriesIndex
            The hourly counts used to compute the time trend
            used for later scaling
        y : Ignored
        """
        if not self.remove_trend:
            return self

        # drop possibly incomplete final month
        if X.index.max().day < 25:
            lastday = (X.index.max() - timedelta(days=X.index.max().day - 1)).date()
            X = X[:lastday][:-1]
        series = X.resample('M').mean().fillna(value=1)
        # reset index to middle of month
        series.index = series.index - timedelta(days=15)
        # decompose target into time trend and seasonal variation
        result = seasonal_decompose(series, freq=12,
                                    model='multiplicative',
                                    extrapolate_trend=True)

        trend = result.trend.reindex(X.index).interpolate(
            method='linear', limit_area='inside')
        # fit trend with linear function
        self.trend_model_ = LinearRegression(fit_intercept=True).fit(
            self._datetime_to_numpy(trend.dropna().index).reshape(-1, 1),
            trend.dropna())

        # calculate norm to normalize trend.
        # not strictly necessary, but preserves count order of magnitude
        self.norm = self.trend_model_.predict(
            self._datetime_to_numpy(X.index).reshape(-1, 1)).ravel()[0]
        return self

    def _datetime_to_numpy(self, time_index):
        return np.array(time_index.astype(np.int64) / 10e16)

    def transform(self, X, y='deprecated'):
        """Perform standardization by subtracting the time trend
        (if subtract_trend is True)

        Parameters
        ----------
        X : Series with TimeSeriesIndex contains counts to be scaled
        y : Ignored

        Returns
        -------
        Transformed Series with TimeSeriesIndex
        """
        if not self.remove_trend:
            return X

        X_tr = X.copy(deep=True)
        trend = self.trend_model_.predict(
            self._datetime_to_numpy(X.index).reshape(-1, 1))
        X_tr = X_tr / (trend / self.norm)
        return X_tr

    def inverse_transform(self, X, y='deprecated'):
        if not self.remove_trend:
            return X

        X_tr = X.copy(deep=True)
        trend = self.trend_model_.predict(
            self._datetime_to_numpy(X.index).reshape(-1, 1))
        X_tr = X_tr * (trend / self.norm)
        return X_tr


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical=False, remove_year=True):
        """
        Transform features of the bike sharing data set to be used as input
        for BikeshareRegression.

        Real-valued featues are min-max transformed.

        The fit and transform member functions operate on a pandas DataFrame with TimeSeriesIndex.

        Parameters
        ----------
        remove_year : if True, assume that the yearly trend was already removed
                      e.g. by TrendRemover, so we can remove the year and avoid
                      overfitting on it.

        categorical : if True, split ordinal features into indicator variables
        """
        self.remove_year = remove_year
        self.categorical = categorical

    def transform(self, X, y='deprecated'):
        # encode categorical features
        if self.categorical:
            X_enc = self.onehotencoder_.fit_transform(X[self.catcols]).toarray()
        else:
            X_enc = np.asarray(X[self.catcols])

        # scale real features
        X_minmax = self.minmaxscaler_.transform(X[self.realcols_])

        # reset time index
        X_tr = pd.DataFrame(np.hstack((X_enc, X_minmax, X[self.indicatorcols_])))
        X_tr = X_tr.set_index(X.index)
        return X_tr

    def fit(self, X, y=None):
        self.realcols_ = ['temp', 'atemp', 'hum', 'windspeed']
        self.indicatorcols_ = ['workingday', 'holiday']
        if not self.remove_year:
            self.indicatorcols_.append('yr')
        season = list(range(1, 5))
        month = list(range(1, 13))
        hour = list(range(24))
        weekday = list(range(7))
        weathersit = list(range(1, 5))
        categories = [season, month, hour, weekday, weathersit]
        self.catcols = ['season', 'mnth', 'hr', 'weekday', 'weathersit']
        self.onehotencoder_ = OneHotEncoder(categories=categories)
        self.minmaxscaler_ = MinMaxScaler().fit(X[self.realcols_])
        return self
