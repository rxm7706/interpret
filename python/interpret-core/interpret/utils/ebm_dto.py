import json
import pkg_resources

from ..glassbox.ebm.ebm import (BaseCoreEBM, EBMExplanation, 
    ExplainableBoostingClassifier, EBMPreprocessor)


class LearnerDTO:
    def __init__(self, feature_names, feature_types):
        self.feature_names = feature_names
        self.feature_types = feature_types

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.feature_names == other.feature_names and
            self.feature_types == other.feature_types
        )

    def __hash__(self):
        return hash((tuple(self.feature_names), 
            tuple(self.feature_types)))

    @classmethod
    def from_json(cls, data):
        return cls(**data)


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
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True,
            indent=4)

    @classmethod
    def from_ebm(cls, ebm):
        version = None
        learner = None
        
        if ebm is not None:
            version = list(map(int, 
                pkg_resources.get_distribution("interpret-core").version.split('.')))
            learner = LearnerDTO(ebm.feature_names, ebm.feature_types)
        return cls(version, learner)

    @classmethod
    def from_json(cls, data):
        version = data["version"]
        learner = LearnerDTO.from_json(data["learner"])
        return cls(version, learner)



