import pandas as pd
import numpy as np
import json

from ..glassbox.ebm.ebm import EBMExplanation
from json import JSONEncoder, JSONDecoder

class ExplanationJSONEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, EBMExplanation):
            return {
                "_type": "EBMExplanation",
                "value": obj.__dict__
            }
        elif isinstance(obj, np.ndarray):
            return {
                "_type": "np.ndarray",
                "value": obj.tolist(),
            }
        elif isinstance(obj, np.int32):
            return {
                "_type": "np.int32",
                "value": int(obj)
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
        elif _type == "np.int32":
            return np.int32(obj["value"])
        elif _type == "np.int64":
            return np.int64(obj["value"])
        elif _type == "pd.DataFrame":
            return pd.read_json(obj["value"])
        return obj


def deserialize_explanation(explanation_dict):
    return EBMExplanation(
        explanation_dict["explanation_type"],
        explanation_dict["_internal_obj"],
        explanation_dict["feature_names"],
        explanation_dict["feature_types"],
        explanation_dict["name"],
        explanation_dict["selector"]
    )

def to_json(explanation, output_file):
    with open(output_file, "w") as file:
        json.dump(explanation, file, indent=4, cls=ExplanationJSONEncoder)

def from_json(json_file):
    with open(json_file, "r") as file:
        return json.load(file, cls=ExplanationJSONDecoder)
