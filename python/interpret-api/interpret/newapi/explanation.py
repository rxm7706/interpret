import json

from slicer import Slicer as S
from interpret.newapi.component import Component
from interpret.newapi.serialization import ExplanationJSONDecoder, ExplanationJSONEncoder


class Explanation(S):
    @classmethod
    def _init_explanation(cls, instance, *args):
        super(Explanation, instance).__init__()

        instance.components = {}
        instance._field_components_map = {}

        for value in args:
            if value is not None:
                instance.append(value)

    def __init__(self, **kwargs):
        self.__class__._init_explanation(self, *list(kwargs.values()))

    def append(self, component):
        if not isinstance(component, Component):
            raise Exception(f"Can't append object of type {type(component)} to this object.")

        self.components[type(component)] = component
        for field_name, field_value in component.fields.items():
            self.__setattr__(field_name, field_value)
            self._field_components_map[field_name] = type(component)

        return self

    def __contains__(self, item):
        return item in self.components

    def __repr__(self):
        record = self.components.copy()
        record = {str(key.__name__): str(list(val.fields.keys())) for key, val in record.items()}
        class_name = self.__class__.__name__
        attributes = json.dumps(record, indent=2)

        return f"{class_name}:\n{attributes}"

    @classmethod
    def from_json(cls, json_str):
        d = json.loads(json_str, cls=ExplanationJSONDecoder)
        instance = d["content"]
        return instance

    @classmethod
    def from_components(cls, components):
        instance = cls.__new__(cls)
        cls._init_explanation(instance, *components)

        return instance

    def to_json(self, **kwargs):
        version = "0.0.1"
        di = {
            "version": version,
            "content": self,
        }

        return json.dumps(di, cls=ExplanationJSONEncoder, **kwargs)


class AttribExplanation(Explanation):
    def __init__(self, attrib, data=None, perf=None, bound=None, **kwargs):
        super().__init__(
            attrib=attrib,
            data=data,
            perf=perf,
            bound=bound,
            **kwargs,
        )
