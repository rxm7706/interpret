# Copyright (c) 2019 Microsoft Corporation
# Distributed under the MIT software license

from ....test.utils import (
    synthetic_multiclass,
    synthetic_classification,
    adult_classification,
    iris_classification,
)
from ..ebm import ExplainableBoostingRegressor, ExplainableBoostingClassifier
from ..ebm import DPExplainableBoostingRegressor, DPExplainableBoostingClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import pytest

import warnings

def test_ebm_regression():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingRegressor()
    clf.fit(X_train, y_train)
    scores = clf.predict(X_test)

    assert(scores[0] == 0.18899612553039272)
    assert(scores[1] == 0.9200925815978228)

    mse = mean_squared_error(y_test, scores)
    assert(mse == 0.048204238779861804)

def test_ebm_binary():
    from sklearn.metrics import roc_auc_score

    data = adult_classification()
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    clf = ExplainableBoostingClassifier()
    clf.fit(X_tr, y_tr)

    prob_scores = clf.predict_proba(X_te)

    assert(prob_scores[0][0] == 0.9894016952780178)
    assert(prob_scores[1][0] == 0.91610229694132)

    auc = roc_auc_score(y_te, prob_scores[:, 1])
    assert(auc == 0.9753265602322206)

def test_ebm_multi():
    from sklearn.metrics import accuracy_score

    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingClassifier()
    clf.fit(X_train, y_train)

    prob_scores = clf.predict_proba(X_test)

    assert(prob_scores[0][0] == 0.9993348289594844)
    assert(prob_scores[1][0] == 0.0017780209082921628)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    assert(accuracy == 0.9666666666666667)

def test_dp_ebm_regression():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    np.random.seed(42) # hack the random number generator to make the same numbers for DP
    clf = DPExplainableBoostingRegressor(n_jobs=1)
    clf.fit(X_train, y_train)
    scores = clf.predict(X_test)

    assert(scores[0] == 0.3362108819837002)
    assert(scores[1] == -0.3825590736213982)

    mse = mean_squared_error(y_test, scores)
    assert(mse == 0.36903169427656024)

def test_dp_ebm_binary():
    from sklearn.metrics import roc_auc_score

    data = adult_classification()
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    np.random.seed(42) # hack the random number generator to make the same numbers for DP
    clf = DPExplainableBoostingClassifier(n_jobs=1)
    clf.fit(X_tr, y_tr)

    prob_scores = clf.predict_proba(X_te)

    assert(prob_scores[0][0] == 0.4862988878825557)
    assert(prob_scores[1][0] == 0.4316063376513217)

    auc = roc_auc_score(y_te, prob_scores[:, 1])
    assert(auc == 0.6632801161103048)



def test_3_way():
    data = iris_classification()
    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    clf = ExplainableBoostingRegressor(interactions=[(0,1,2), (0,1,3), (1,2,3), (0, 1)])
    clf.fit(X_train, y_train)
    scores = clf.predict(X_test)

    assert(scores[0] == 0.17279005286179835)
    assert(scores[1] == 0.9616620874179362)

    mse = mean_squared_error(y_test, scores)
    assert(mse == 0.0464657086398099)

