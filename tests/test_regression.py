from bikeshare.data import load_data
from bikeshare.preprocessing import TrendRemover, FeatureTransformer
from bikeshare.model import BikeshareRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def test_bikeshareregression():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    ft = FeatureTransformer()
    X_train = ft.fit_transform(X_train)
    pr = TrendRemover(remove_trend=True).fit(y_train['cnt'])
    bsr = BikeshareRegression(trend_remover=pr, random_state=42).fit(X_train, y_train['cnt'])

    X_test = ft.transform(X_test)
    y_pred = bsr.predict(X_test)

    # test default regressor, this isn't the best possible result.
    mae = mean_absolute_error(y_test['cnt'], y_pred)
    assert mae < 45
