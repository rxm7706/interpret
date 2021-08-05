import json
from interpret.utils.serialization import from_json
from jsonschema import validate
import pkg_resources
import os

from ..glassbox.ebm.ebm import (BaseCoreEBM, EBMExplanation,
    ExplainableBoostingClassifier, EBMPreprocessor, ExplainableBoostingRegressor)

class ClassificationTaskDTO:
    def __init__(self, classes):
        self.type = 'classification'
        self.classes = classes

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.type == other.type and
            self.classes == other.classes
        )

    def __hash__(self):
        return hash(self.type, tuple(self.classes))

    @classmethod
    def from_json(cls, json_dict):
        return cls(json_dict['classes'])

class RegressionTaskDTO:
    def __init__(self):
        self.type = 'regression'

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.type == other.type
        )

    def __hash__(self):
        return hash(self.type)

    @classmethod
    def from_json(cls, json_dict):
        return cls()

class InterpretableEBMDTO:
    def __init__(self, task, intercept, additive_terms, standard_devations,
        feature_groups):
        self.task = task
        self.intercept = intercept
        self.additive_terms = additive_terms
        self.standard_deviations = standard_devations
        self.feature_groups = feature_groups

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.task == other.task and
            self.intercept == other.intercept and
            self.additive_terms == other.additive_terms and
            self.standard_deviations == self.standard_deviations and
            self.feature_groups == other.feature_groups
        )

    def __hash__(self):
        hashable_additive_terms = tuple((lambda l: tuple(l), self.additive_terms))
        hashable_standard_deviations = tuple((lambda l: tuple(l), self.standard_deviations))

        return hash(
            self.task,
            self.intercept,
            hashable_additive_terms,
            hashable_standard_deviations,
            self.feature_groups
        )

    @classmethod
    def from_json(cls, json_dict):
        task_dict = json_dict["task"]
        task = None

        if task_dict["type"] == "regression":
            task = RegressionTaskDTO.from_json(task_dict)
        else:
            task = ClassificationTaskDTO.from_json(task_dict)

        feature_groups = FeatureGroupDTO.from_json(json_dict['feature_groups'])

        return cls(
            task,
            json_dict["intercept"],
            json_dict["additive_terms"],
            json_dict["standard_deviations"],
            feature_groups)

class FeatureGroupDTO:
    def __init__(self, groups, importances, mins, maxes, bin_edges):
        self.groups = groups
        self.importances = importances
        self.mins = mins
        self.maxes = maxes
        self.bin_edges = bin_edges

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
            self.groups == other.groups and \
            self.importances == other.importances and \
            self.mins == other.mins and \
            self.maxes == other.maxes and \
            self.bin_edges == other.bin_edges

    def __hash__(self):
        hashable_bin_edges = tuple((lambda l: tuple(l), self.bin_edges))

        return hash((tuple(self.groups),
            tuple(self.importances),
            tuple(self.mins),
            tuple(self.maxes),
            tuple(hashable_bin_edges)))

    @classmethod
    def from_json(cls, json_dict):
        mins = {
            int(k): v for k, v in json_dict["mins"].items()
        }
        maxes = {
            int(k): v for k, v in json_dict["maxes"].items()
        }
        bin_edges = {
            int(k): v for k, v in json_dict["bin_edges"].items()
        }

        return cls(
            json_dict["groups"],
            json_dict["importances"],
            mins,
            maxes,
            bin_edges)

class LearnerDTO:
    def __init__(self, feature_names, feature_types, interpretable_ebm):
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.interpretable_ebm = interpretable_ebm

    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
            self.feature_names == other.feature_names and \
            self.feature_types == other.feature_types and \
            self.interpretable_ebm == other.interpretable_ebm

    def __hash__(self):
        return hash((tuple(self.feature_names),
            tuple(self.feature_types),
            self.interpretable_ebm))

    @classmethod
    def from_json(cls, json_dict):
        feature_names = json_dict["feature_names"]
        feature_types = json_dict["feature_types"]
        intepretable_ebm = InterpretableEBMDTO.from_json(json_dict["interpretable_ebm"])
        return cls(feature_names, feature_types, intepretable_ebm)

class EBMDTO:
    def __init__(self, version, learner):
        self.version = version
        self.learner = learner

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.version == other.version and
            self.learner == other.learner
        )

    def __hash__(self):
        return hash((tuple(self.version),
            self.learner))

    def to_ebm(self):
        raise NotImplementedError

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)

    @classmethod
    def from_ebm(cls, ebm):
        version = None
        learner = None
        interpretable_ebm = None

        if ebm is not None:
            version = list(map(int,
                pkg_resources.get_distribution("interpret-core").version.split('.')))

            task = None
            if type(ebm) is ExplainableBoostingClassifier:
                task = ClassificationTaskDTO(ebm.classes_.tolist())
            elif type(ebm) is ExplainableBoostingRegressor:
                task = RegressionTaskDTO()

            intercept = ebm.intercept_[0].item()
            additive_terms = list(map(lambda a: a.tolist(), ebm.additive_terms_))
            standard_deviations = list(map(lambda a: a.tolist(), ebm.term_standard_deviations_))

            feature_groups_groups = ebm.feature_groups_
            feature_groups_importances = list(map(lambda i: i.item(), ebm.feature_importances_))
            feature_groups_mins = ebm.preprocessor_.col_min_
            feature_groups_maxes = ebm.preprocessor_.col_max_
            feature_groups_bin_edges = {
                k: v.tolist() for k, v in ebm.preprocessor_.col_bin_edges_.items()
            }

            feature_groups = FeatureGroupDTO(
                feature_groups_groups,
                feature_groups_importances,
                feature_groups_mins,
                feature_groups_maxes,
                feature_groups_bin_edges
            )

            interpretable_ebm = InterpretableEBMDTO(
                task,
                intercept,
                additive_terms,
                standard_deviations,
                feature_groups
            )

            learner = LearnerDTO(ebm.feature_names, ebm.feature_types, interpretable_ebm)

        return cls(version, learner)

    @classmethod
    def from_json(cls, json_dict):
        version = json_dict["version"]
        learner = LearnerDTO.from_json(json_dict["learner"])
        return cls(version, learner)

    @classmethod
    def load_json(cls, json_str):
        json_dict = json.loads(json_str)

        cur_dir = os.path.dirname(__file__)
        with open(os.path.join(cur_dir, 'ebm.schema.json')) as json_file:
            schema = json.load(json_file)
            validate(instance=json_dict, schema=schema)

        return EBMDTO.from_json(json_dict)
