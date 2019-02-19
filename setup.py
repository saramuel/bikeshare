from setuptools import setup, find_packages

setup(
    name = 'bikeshare',
    version = '1.0.0',
    url = 'https://github.com/saramuel/bikeshare.git',
    author = 'Sarah Mueller',
    author_email = 'sarah.mueller@gmx.fr',
    description = 'Prediction model for the hourly utilization “cnt” of this data set:https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset',
    packages = find_packages(),    
    install_requires = ['scikit-learn','pandas','matplotlib'],
)
