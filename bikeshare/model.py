import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class BikeshareRegression(BaseEstimator, RegressorMixin):
    def __init__(self, trend_remover, regressor=None, random_state=None):
        """
        Regression for bikeshare data set.

        Before being passed to the regressor, targets are transformed using a TrendRemover,
        while features are transformed by a FeatureTransformer. The regressor prediction is
        then inverse-transformed through the TrendRemover and cut off at 0.

        The fit, predict, and transform member functions operate on pandas objects
        with TimeSeriesIndex.

        Parameters
        ----------
        trend_remover : Object of class TrendRemover

        regressor : Regressor object to be used for regression. If None, use RandomForestRegressor

        random_state : Seed for random number generator used in regression.
        """
        if regressor is None:
            regressor = RandomForestRegressor(n_estimators=10, random_state=random_state)
        self.trend_remover = trend_remover
        self.regressor = regressor

    def fit(self, X, y, sample_weight=None):
        y = self.trend_remover.transform(y)
        self.regressor.fit(X, y)
        return self

    def predict(self, X):
        y = pd.Series(self.regressor.predict(X))
        y.index = X.index
        return np.maximum(0, self.trend_remover.inverse_transform(y))

    def score(self, X, y):
        # score is negative error
        return -mean_absolute_error(y, self.predict(X))


class CombinedModel(BaseEstimator, RegressorMixin):
    def __init__(self, casual_reg, registered_reg):
        """
        Combine BikeShareRegression models for casual and registered users by
        adding their predictions.

        Parameters
        ----------
        casual_reg : BikeshareRegression object for casual users

        registered_reg : BikeshareRegression object for registered users
        """
        self.casual_reg = casual_reg
        self.registered_reg = registered_reg

    def fit(self, X, y):
        self.casual_reg.fit(X, y['casual'])
        self.registered_reg.fit(X, y['registered'])
        return self

    def predict(self, X):
        series = self.casual_reg.predict(X) + self.registered_reg.predict(X)
        series.name = 'cnt'
        return np.maximum(0, series)
