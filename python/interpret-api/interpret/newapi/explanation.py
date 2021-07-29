import json

from slicer import Slicer as S
from interpret.newapi.component import Component


class Explanation:
    @classmethod
    def _init_explanation(cls, instance, *args):
        super(Explanation, instance).__init__()

        instance._slicer = S()
        instance._field_component_map = {}
        instance._component_fields_map = {}

        for value in args:
            if value is not None:
                instance._append(value)

    def __init__(self, **kwargs):
        self.__class__._init_explanation(self, *list(kwargs.values()))

    def __getitem__(self, item):
        new_explanation = Explanation.from_components(self._components().values())
        new_explanation._slicer = self._slicer.__getitem__(item)
        return new_explanation

    def __getattr__(self, item):
        if item.startswith("_"):  # Protected attribute
            return super(Explanation, self).__getattr__(item)
        else:  # Public attribute
            return self._slicer.__getattr__(item)

    def _components(self):
        updated_components = {}
        for component_type, fields in self._component_fields_map.items():
            field_map = {field: self.__getattr__(field) for field in fields}
            updated_component = component_type(**field_map)
            updated_components[component_type] = updated_component
        return updated_components

    def _append(self, component):
        if not isinstance(component, Component):
            raise Exception(f"Can't append object of type {type(component)} to this object.")

        component_key = type(component)
        self._component_fields_map[component_key] = []
        for field_name, field_value in component.fields.items():
            self._slicer.__setattr__(field_name, field_value)
            self._field_component_map[field_name] = component_key
            self._component_fields_map[component_key].append(field_name)
        return self

    def __contains__(self, item):
        return item in self._component_fields_map.keys()

    def __repr__(self):
        record = self._components().copy()
        fields = []
        shape_str = f"shape: {self.shape}"
        fields.append(shape_str)
        fields.append("-" * len(shape_str))
        for record_key, record_val in record.items():
            for field_name, field_val in record_val.fields.items():
                field_value = str(self.__getattr__(field_name))

                if field_name in self._dims:
                    field_value_str = f"Dim\t{field_name} = {field_value}"
                else:
                    if field_name in self._objects:
                        field_type = 'O'
                        field_dim = ','.join(str(x) for x in self._objects[field_name].dim)
                    else:
                        field_type = 'A'
                        field_dim = ','.join(str(x) for x in self._aliases[field_name].dim)
                    field_value_str = f"{field_type}{{{field_dim}}}\t{field_name} = {field_value}"

                if len(field_value_str) > 60:
                    field_value_str = field_value_str[:57] + "..."
                fields.append(field_value_str)
        fields = "\n".join(fields)

        return fields

    @classmethod
    def from_json(cls, json_str):
        from interpret.newapi.serialization import ExplanationJSONDecoder

        d = json.loads(json_str, cls=ExplanationJSONDecoder)
        instance = d["content"]
        return instance

    @classmethod
    def from_components(cls, components):
        instance = cls.__new__(cls)
        cls._init_explanation(instance, *components)

        return instance

    def to_json(self, **kwargs):
        from interpret.newapi.serialization import ExplanationJSONEncoder
        version = "0.0.1"
        di = {
            "version": version,
            "content": self,
        }

        return json.dumps(di, cls=ExplanationJSONEncoder, **kwargs)


class AttribExplanation(Explanation):
    def __init__(self, attrib, data=None, perf=None, bound=None, meta=None, **kwargs):
        super().__init__(
            attrib=attrib,
            data=data,
            perf=perf,
            bound=bound,
            meta=meta,
            **kwargs,
        )

