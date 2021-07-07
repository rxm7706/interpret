from ...glassbox.ebm.ebm import ExplainableBoostingClassifier
from ..serialization import ExplanationJSONEncoder, ExplanationJSONDecoder, exp_from_json, exp_to_json
from ...test.utils import adult_classification

import numpy as np
from sklearn.model_selection import (
    cross_validate,
    StratifiedShuffleSplit,
    train_test_split,
)

import pytest

def inspect_global_exp(global_exp, file_name):
    import os
    import pprint
    import inspect

    output_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_path, file_name)

    with open(output_path, "w") as f:
        pprint.pprint(inspect.getmembers(global_exp), stream=f)

def test_explanation_serialization():

    data = adult_classification()
    X = data["full"]["X"]
    y = data["full"]["y"]
    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=3)
    clf.fit(X_tr, y_tr)

    global_exp = clf.explain_global()

    import os

    output_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_path, "expToJson.json")

    exp_to_json(global_exp, output_path)
    
    new_exp = exp_from_json(output_path)

    #inspect_global_exp(global_exp, "global_exp.txt")
    #inspect_global_exp(new_exp, "new_exp.txt")
