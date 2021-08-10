import json
from interpret.utils.serialization import from_json
from jsonschema import validate
import numpy as np
from numpy.lib.histograms import histogram_bin_edges
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
    def __init__(self, task, intercept, additive_terms, standard_deviations,
        interactions, feature_groups):
        self.task = task
        self.intercept = intercept
        self.additive_terms = additive_terms
        self.standard_deviations = standard_deviations
        self.interactions = interactions
        self.feature_groups = feature_groups

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.task == other.task and
            self.intercept == other.intercept and
            self.additive_terms == other.additive_terms and
            self.standard_deviations == self.standard_deviations and
            self.interactions == self.interactions and
            self.feature_groups == other.feature_groups
        )

    def __hash__(self):
        hashable_additive_terms = tuple(map(tuple, self.additive_terms))
        hashable_standard_deviations = tuple(map(tuple, self.standard_deviations))

        return hash(
            self.task,
            self.intercept,
            hashable_additive_terms,
            hashable_standard_deviations,
            self.interactions,
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
            json_dict["interactions"],
            feature_groups)

class FeatureGroupDTO:
    def __init__(self, groups, importances, mins, maxes, bin_edges, hist_edges,
        hist_counts):
        self.groups = groups
        self.importances = importances
        self.mins = mins
        self.maxes = maxes
        self.bin_edges = bin_edges
        self.hist_edges = hist_edges
        self.hist_counts = hist_counts


    def __eq__(self, other):
        return self.__class__ == other.__class__ and \
            self.groups == other.groups and \
            self.importances == other.importances and \
            self.mins == other.mins and \
            self.maxes == other.maxes and \
            self.bin_edges == other.bin_edges and \
            self.hist_edges == other.hist_edges and \
            self.hist_counts == other.hist_counts

    def __hash__(self):
        hashable_bin_edges = tuple(map(tuple, self.bin_edges))
        hashable_hist_edges = tuple(map(tuple, self.hist_edges))
        hashable_hist_counts = tuple(map(tuple, self.hist_counts))

        return hash((tuple(self.groups),
            tuple(self.importances),
            tuple(self.mins),
            tuple(self.maxes),
            hashable_bin_edges,
            hashable_hist_edges,
            hashable_hist_counts))

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
        hist_edges = {
            int(k): v for k, v in json_dict["hist_edges"].items()
        }
        hist_counts = {
            int(k): v for k, v in json_dict["hist_counts"].items()
        }

        return cls(
            json_dict["groups"],
            json_dict["importances"],
            mins,
            maxes,
            bin_edges,
            hist_edges,
            hist_counts)

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
        ebm = None
        classes = None

        if type(self.learner.interpretable_ebm.task) == ClassificationTaskDTO:
            # BUGBUG: This initializes the object with a bunch of default meta
            # params that haven't yet been serialized
            ebm = ExplainableBoostingClassifier()
            classes = np.array(self.learner.interpretable_ebm.task.classes)
        else:
            ebm = ExplainableBoostingRegressor()

        additive_terms = list(
            map(lambda a: np.array(a),
                self.learner.interpretable_ebm.additive_terms)
        )
        standard_deviations = list(
            map(lambda a: np.array(a),
                self.learner.interpretable_ebm.standard_deviations))

        ebm.additive_terms_ = additive_terms
        ebm.classes_ = classes
        ebm.feature_names = self.learner.feature_names
        ebm.feature_types = self.learner.feature_types
        ebm.feature_groups_ = self.learner.interpretable_ebm.feature_groups.groups
        ebm.feature_importances_ = list(
            map(lambda i: np.float64(i),
                self.learner.interpretable_ebm.feature_groups.importances)
        )
        ebm.interactions = self.learner.interpretable_ebm.interactions
        ebm.intercept_ = np.array([self.learner.interpretable_ebm.intercept])
        ebm.term_standard_deviations_ = standard_deviations

        col_bin_edges = {
            k: np.array(v) for k, v
            in self.learner.interpretable_ebm.feature_groups.bin_edges.items()
        }

        hist_edges = {
            k: np.array(v) for k, v
            in self.learner.interpretable_ebm.feature_groups.hist_edges.items()
        }

        hist_counts = {
            k: np.array(v) for k, v
            in self.learner.interpretable_ebm.feature_groups.hist_counts.items()
        }

        ebm.preprocessor_ = EBMPreprocessor(self.learner.feature_names,
            self.learner.feature_types)
        ebm.preprocessor_.col_bin_edges_ = col_bin_edges
        ebm.preprocessor_.col_max_ = self.learner.interpretable_ebm.feature_groups.maxes
        ebm.preprocessor_.col_min_ = self.learner.interpretable_ebm.feature_groups.mins
        ebm.preprocessor_.col_types_ = self.learner.feature_types
        ebm.preprocessor_.hist_edges_ = hist_edges
        ebm.preprocessor_.hist_counts_ = hist_counts

        ebm.preprocessor_.has_fitted_ = True
        ebm.has_fitted_ = True

        return ebm

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
            interactions = ebm.interactions

            feature_groups_groups = ebm.feature_groups_
            feature_groups_importances = list(map(lambda i: i.item(), ebm.feature_importances_))
            feature_groups_mins = ebm.preprocessor_.col_min_
            feature_groups_maxes = ebm.preprocessor_.col_max_
            feature_groups_bin_edges = {
                k: v.tolist() for k, v in ebm.preprocessor_.col_bin_edges_.items()
            }
            feature_groups_hist_edges = {
                k: v.tolist() for k, v in ebm.preprocessor_.hist_edges_.items()
            }
            feature_groups_hist_counts = {
                k: v.tolist() for k, v in ebm.preprocessor_.hist_counts_.items()
            }

            feature_groups = FeatureGroupDTO(
                feature_groups_groups,
                feature_groups_importances,
                feature_groups_mins,
                feature_groups_maxes,
                feature_groups_bin_edges,
                feature_groups_hist_edges,
                feature_groups_hist_counts
            )

            interpretable_ebm = InterpretableEBMDTO(
                task,
                intercept,
                additive_terms,
                standard_deviations,
                interactions,
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
