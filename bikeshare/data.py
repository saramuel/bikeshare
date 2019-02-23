import pandas as pd
import os.path


def load_data(data_dir=None):
    """
    load bikeshare data.

    data_dir: directory containing hour.csv
    """
    if data_dir is None:
        package_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir))
        data_dir = os.path.join(package_dir, 'dataset')
    df_hour = pd.read_csv(os.path.join(data_dir, 'hour.csv'), sep=',')
    df_hour['dteday'] = (pd.to_datetime(df_hour['dteday'])
                         + pd.to_timedelta(df_hour['hr'], unit='hour'))
    df_hour = df_hour.set_index('dteday')
    X = (df_hour.drop('cnt', axis=1)
         .drop('instant', axis=1)
         .drop('registered', axis=1)
         .drop('casual', axis=1))
    y = df_hour[['casual', 'registered', 'cnt']]
    return X, y
