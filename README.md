Synopsis
========

Build a model for the [Bike Sharing Dataset Data Set](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset).

The hyper-paramters are cross-validated on a TimeSeriesSplit and the final
model is evaluated on a test set comprising the last 20% of the data points.

Only a linear regression and random forest regression are attempted.

Pipeline
========

1. split dataset into casual and registered users, predict both totals separately
2. transform target
  1. remove multiplicative linear trend in time from target to account for customer growth
  2. log-transform the target
3. transform features
  1. split categorical features into binary indicators (cross-validated whether this helps)
  2. min-max transform real-valued features
  3. remove year feature (cross-validated whether this helps), as trend already took this into account
4. learn regressor for transformed target on transformed features, and undo transformations on prediction
5. add regressor predictions for casual/registered

Performance
===========

The mean absolute error on the test set is 36.83.
The final model is obtained from learning on the complete data set.
It is saved in {model-output-dir}/final.model.

Requirements
============

Python 3.6

Installation
============

    git clone https://github.com/saramuel/bikeshare.git
    cd bikeshare

    # if conda is available, create environment via:
    conda env create -f environment.yml
    conda activate bikeshare

    # otherwise use virtualenv etc.
    # ...

    pip install -e .

Running Tests
=============

    # install test dependencies
    pip install -e '.[dev]'

    # run tests
    pytest --flake8

Running
=======

    ./scripts/bikeshare {data-dir} {model-output-dir} {plots-output-dir}

    # grid search results are cached in .joblib_cache directory, delete to fit again.
