import pandas as pd
import numpy as np
import json

from ..glassbox.ebm.ebm import EBMExplanation
from json import JSONEncoder, JSONDecoder

class ExplanationJSONEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "_type": "np.ndarray",
                "value": obj.tolist(),
            }
        elif isinstance(obj, EBMExplanation):
            return {
                "_type": "EBMExplanation",
                "value": obj.__dict__
            }
        elif isinstance(obj, np.int64):
            return {
                "_type": "np.int64",
                "value": int(obj)
            }
        elif isinstance(obj, pd.DataFrame):
            return {
                "_type": "pd.DataFrame",
                "value": obj.to_json()
            }
        else:
            return JSONEncoder.default(self, obj)


class ExplanationJSONDecoder(JSONDecoder):

    def __init__(self, *args, **kwargs):
        JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if "_type" not in obj:
            return obj

        _type = obj["_type"]

        if _type == "EBMExplanation":
            return deserialize_explanation(obj["value"])
        elif _type == "np.ndarray":
            return np.array(obj["value"])
        elif _type == "np.int64":
            return np.int64(obj["value"])
        elif _type == "pd.DataFrame":
            return pd.read_json(obj["value"])
        return obj


def serialize_explanation(explanation):
    serializable_exp = {
        "version" : "0.0.1",
        "explanation_type" : explanation.explanation_type,
        "internal_obj": explanation._internal_obj,
        "feature_names": explanation.feature_names,
        "feature_types": explanation.feature_types,
        "name": explanation.name,
        "selector": explanation.selector
    }
    return serializable_exp


def deserialize_explanation(explanation_dict):
    return EBMExplanation(
        explanation_dict["explanation_object"],
        explanation_dict["_internal_obj"],
        explanation_dict["feature_names"],
        explanation_dict["feature_types"],
        explanation_dict["name"],
        explanation_dict["selector"]
    )

def exp_to_json(explanation, output_json):
    with open(output_json, "w") as output_file:
        json.dump(explanation, output_file, indent=4, cls=ExplanationJSONEncoder)

def exp_from_json(json_file):
    with open(json_file, "r") as output_json:
        return json.load(output_json, cls=ExplanationJSONDecoder)
