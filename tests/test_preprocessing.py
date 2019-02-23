# import matplotlib.pyplot as plt
import numpy as np

from bikeshare.preprocessing import TrendRemover
from bikeshare.preprocessing import FeatureTransformer
from bikeshare.data import load_data


def test_loading():
    X, y = load_data()
    assert 'cnt' not in X.columns
    assert 'registered' not in X.columns
    assert 'casual' not in X.columns
    assert set(y.columns) == {'cnt', 'registered', 'casual'}
    assert (X.index == y.index).all()


def get_periodical_testset():
    X, y = load_data()
    y_cnt = y['cnt']
    trend = np.linspace(1, 2, len(y_cnt))
    periodical = 2000 * (np.abs(np.sin(
        np.linspace(0, 1, len(y_cnt)) * 2 * np.pi)) + 1)
    y_cnt[:] = trend * periodical
    return y_cnt, periodical, trend


def test_trendremover():
    y, periodical, trend = get_periodical_testset()
    trend_remover = TrendRemover(remove_trend=True)
    y_trans = np.array(trend_remover.fit_transform(y))
    diff = y_trans - periodical
    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(trend, trend*periodical, color='green')
    # ax.scatter(trend, periodical, color='black')
    # ax.scatter(trend, np.array(y_trans).ravel(), color='grey')
    # plt.show()
    assert np.std(diff) < 35


def test_trendremover_real():
    X, y = load_data()
    y = y['cnt']
    trend_remover = TrendRemover(remove_trend=True)
    y_trans = np.array(trend_remover.fit_transform(y))
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(np.linspace(1,2,len(y)), np.array(y).ravel(), color='black')
    # ax.plot(np.linspace(1,2,len(y)), np.array(y_trans).ravel(), color='grey')
    # plt.show()
    assert np.std(y_trans) < 0.5*np.std(y)


def test_featuretransformer():
    X, y = load_data()
    ft = FeatureTransformer(remove_year=True, categorical=False)
    X_tr = ft.fit_transform(X)
    assert len(X_tr.columns) == 11
    assert len(X_tr) == len(X)
    assert (X.index == X_tr.index).all()

    ft = FeatureTransformer(remove_year=False, categorical=False)
    X_tr = ft.fit_transform(X)
    assert len(X_tr.columns) == 12
    assert len(X_tr) == len(X)
    assert (X.index == X_tr.index).all()

    ft = FeatureTransformer(remove_year=True, categorical=True)
    X_tr = ft.fit_transform(X)
    assert len(X_tr.columns) == 57
    assert len(X_tr) == len(X)
    assert (X.index == X_tr.index).all()

    ft = FeatureTransformer(remove_year=False, categorical=True)
    X_tr = ft.fit_transform(X)
    assert len(X_tr.columns) == 58
    assert len(X_tr) == len(X)
    assert (X.index == X_tr.index).all()
