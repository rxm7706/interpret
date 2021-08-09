import json

from ...glassbox.ebm.ebm import ExplainableBoostingClassifier
from ...glassbox.ebm.test.test_ebm import _smoke_test_explanations
from ...test.utils import adult_classification
from ..serialization import from_json, to_json
from ..ebm_dto import EBMDTO

from jsonschema import validate
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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

def test_dto_to_json():
    df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
    header=None)

    df.columns = [
        "sample_code_number", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
        "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin",
        "normal_nucleoli", "mitoses", "class"
    ]

    # drop any rows that have missing values
    df = df[~df.eq('?').any(1)]

    # force bare_nuclei column to int64 data type after dropping '?' values
    df['bare_nuclei'] = df['bare_nuclei'].astype(str).astype(int)

    #print(df.head(n=10).to_string(index=False))

    train_cols = df.columns[1:-1]
    label = df.columns[-1]
    X = df[train_cols]
    y = df[label].apply(lambda x: 0 if x == 2 else 1)

    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    ebm_orig = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1, interactions=0)
    ebm_orig.fit(X_train, y_train)   #Works on dataframes and numpy arrays
    ebm_orig_predictions = ebm_orig.predict(X_test)
    ebm_orig_probabilities = ebm_orig.predict_proba(X_test)

    ebm_dto_orig = EBMDTO.from_ebm(ebm_orig)
    json_str = ebm_dto_orig.to_json()
    ebm_dto_deserialized = EBMDTO.load_json(json_str)
    assert(ebm_dto_orig == ebm_dto_deserialized)

    ebm_deserialized = ebm_dto_deserialized.to_ebm()
    ebm_deserialized_predictions = ebm_deserialized.predict(X_test)
    ebm_deserialized_probabilities = ebm_deserialized.predict_proba(X_test)

    assert np.array_equal(ebm_orig_predictions, ebm_deserialized_predictions)
    assert np.array_equal(ebm_orig_probabilities, ebm_deserialized_probabilities)


def test_json_schema_validation():
    schema_str = \
    '{ \
        "$schema": "https://json-schema.org/draft/2019-09/schema", \
        "type": "object", \
        "properties": { \
            "version": { \
                "type": "array", \
                "items": [ \
                { \
                    "type": "number", \
                    "minimum": 0 \
                }, \
                { \
                    "type": "number", \
                    "minimum": 0 \
                }, \
                { \
                    "type": "number", \
                    "minimum": 0 \
                } \
                ], \
                "minItems": 3, \
                "maxItems": 3 \
            }, \
            "learner": { \
                "type": "object", \
                "properties": { \
                    "feature_names": { \
                        "type": "array", \
                        "items": { \
                            "type": "string" \
                        } \
                    }, \
                    "feature_types": { \
                        "type": "array", \
                        "items": { \
                            "type": "string", \
                            "enum": ["continuous", "categorical"] \
                        } \
                    } \
                } \
            } \
        }, \
        "required": [ \
            "version", \
            "learner" \
        ] \
    }'

    schema = json.loads(schema_str)
    myjson = {
        'learner': {
            'feature_names': ['Hola', 'mundo'],
            'feature_types': ['continuous', 'continuous']
        },
        'version': [0, 2, 6]
    }

    validate(myjson, schema)

