from typing import Optional

# import zfit
import marshmallow
from marshmallow import fields, post_dump, post_load


class ZfitBaseSchema(marshmallow.Schema):

    @post_dump
    def post_dump1(self, data, many):
        return {
            key: value for key, value in data.items()
            if value is not None
        }


    @post_dump
    def func2(self, data, many):
        print("func2", data)
        return data


class Parameter:
    def __init__(self, name, value, stepsize=None):
        self.name = name
        self.value = value
        self.stepsize = stepsize
        self.arbitrary = Attribute()


class Attribute:
    def __init__(self, x=None):
        self.x = x


class AttributeRepr(ZfitBaseSchema):
    x = fields.Float(required=False, allow_none=True)

    @post_load
    def make_object(self, data, **kwargs):
        return Attribute(**data)

    def func2(self, data, many):
        print("func2 OTHER OVERLOAD", data)
        return data


class ParameterRepr(ZfitBaseSchema):
    name = fields.Str(required=True)
    value = fields.Float(required=True)
    stepsize = fields.Float(required=False, allow_none=True)
    arbitrary = fields.Nested(AttributeRepr, required=False, allow_none=True)

    @post_load
    def make_object(self, data, **kwargs):
        arbitrary = data.pop("arbitrary")
        param = Parameter(**data)
        param.arbitrary = arbitrary
        return param



param1 = Parameter("param1", 1.0, 0.1)
param2 = Parameter("param2", 2.0)

param_scheme = ParameterRepr()
param1_repr = param_scheme.dump(param1)
param2_repr = param_scheme.dump(param2)

print("Rerpresentation of params")
print(param1_repr)
print(param2_repr)

print("JSON of params")
json1 = param_scheme.dumps(param1)
print(json1)
json2 = param_scheme.dumps(param2)
print(json2)

print("Repr of params from JSON")
param1_repr2 = param_scheme.loads(json1)
param2_repr2 = param_scheme.loads(json2)
print(param1_repr2)
print(param2_repr2)

print("Params from repr")
param1_2 = param_scheme.load(param1_repr)
param2_2 = param_scheme.load(param2_repr)
print(param1_2)
print(param2_2)
