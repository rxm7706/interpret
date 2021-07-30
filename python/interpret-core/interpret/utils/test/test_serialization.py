from ...glassbox.ebm.ebm import ExplainableBoostingClassifier
from ...glassbox.ebm.test.test_ebm import _smoke_test_explanations
from ...test.utils import adult_classification
from ..serialization import from_json, to_json

import numpy as np
import os

# This will be removed later
def inspect_object(object, file_name):
    import os
    import pprint
    import inspect

    output_path = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_path, file_name)

    with open(output_path, "w") as f:
        pprint.pprint(inspect.getmembers(object), stream=f)

def test_classification_serialization():
    data = adult_classification()

    X_tr = data["train"]["X"]
    y_tr = data["train"]["y"]
    X_te = data["test"]["X"]
    y_te = data["test"]["y"]

    clf = ExplainableBoostingClassifier(n_jobs=-2, interactions=3)
    clf.fit(X_tr, y_tr)

    # Serialize classifier after training
    output_path = os.path.dirname(os.path.abspath(__file__))
    classifier_output_path = os.path.join(output_path, "ebmClassifier.json")

    to_json(clf, classifier_output_path)

    inspect_object(clf, "ebmClassifierObject.txt")

    # Create explanations
    global_exp = clf.explain_global()
    local_exp = clf.explain_local(X_te[:5, :], y_te[:5])

    # Serialize explanations
    global_output_path = os.path.join(output_path, "globalExp.json")
    local_output_path = os.path.join(output_path, "localExp.json")

    to_json(global_exp, global_output_path)
    to_json(local_exp, local_output_path)

    # Deserialization of a classifier is not implemented and should raise an error
    #classifier = from_json(classifier_output_path)

    # Deserialize explanations
    global_exp_from_json = from_json(global_output_path)
    local_exp_from_json = from_json(local_output_path)

    _smoke_test_explanations(global_exp, local_exp, 6000)
    _smoke_test_explanations(global_exp_from_json, local_exp_from_json, 6000)

    inspect_object(global_exp, "global_exp.txt")
    inspect_object(global_exp_from_json, "new_global_exp.txt")
