import functools
import pickle
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import Optional, Union, Annotated, Literal, Type, Any

# import zfit
import pydantic
from pydantic import constr, Field, validator, root_validator

reprs = {}


def convert_to_mro(init):
    if isinstance(init, Mapping):
        for k, v in init.items():
            if not isinstance(v, (Iterable, Mapping)):
                continue
            elif 'type' in v:
                type_ = v['type']
                init[k] = reprs[type_](**v).to_orm()
            else:
                init[k] = convert_to_mro(v)

    elif isinstance(init, (list, tuple)):
        init = type(init)([convert_to_mro(v) for v in init])
    return init


def to_mro_init(func):
    @functools.wraps(func)
    def wrapper(self, init, **kwargs):
        init = convert_to_mro(init)
        return func(self, init, **kwargs)

    return wrapper


class MODES(Enum):
    orm = 'orm'
    repr = 'repr'


class BaseRepr(pydantic.BaseModel):
    _constructor = pydantic.PrivateAttr()
    _context = pydantic.PrivateAttr()
    dictionary: Optional[dict] = Field(alias='dict')
    tags: Optional[list[str]]

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True
        allow_population_by_field_name = True
    @classmethod
    def orm_mode(cls):
        return cls._context == MODES.orm

    @classmethod
    def from_orm(cls: Type['Model'], obj: Any) -> 'Model':
        cls._context = MODES.orm
        out = super().from_orm(obj)
        cls._context = MODES.repr
        return out

    def to_orm(self):
        print("Starting to_orm")
        type(self)._context = MODES.orm
        if self._constructor is None:
            raise ValueError("No constructor registered!")
        init = self.dict(exclude_none=True)
        print(init)
        type_ = init.pop('type')
        assert type_ == self.type
        out = self._to_orm(init)
        type(self)._context = MODES.repr
        print(f"Finished to_orm, mode = {self._context}")
        return out

    @to_mro_init
    def _to_orm(self, init):
        return self._constructor(**init)


class Parameter:
    # type = None

    def __init__(self, name, value, stepsize=None):
        self.name = name
        self.value = value
        self.stepsize = stepsize
        self.arbitrary = Attribute()

    def __repr__(self):
        return f"Parameter({self.name}, {self.value}, {self.stepsize})"


class Parameter2:
    type: Literal['Parameter2'] = 'Parameter2'

    def __init__(self, name, value, stepsize=None):
        self.name = name
        self.value = value
        self.stepsize = stepsize
        self.arbitrary = Attribute()
        self._otherattr = 1

    def __repr__(self):
        return f"Parameter2({self.name}, {self.value}, {self.stepsize})"


class PDF:
    type: Literal['PDF'] = 'PDF'

    def __init__(self, name, a, b):
        self.name = name
        self.params = [a, b]

    def __repr__(self):
        return f"PDF({self.name}, {self.params[0]}, {self.params[1]})"


class Attribute:
    def __init__(self, x=None):
        self.x = x


class AttributeRepr(BaseRepr):
    _constructor = Attribute
    x: Optional[float]


class ParameterRepr(BaseRepr):
    _constructor = Parameter
    type: Literal["Parameter"] = "Parameter"
    name: str
    value: float
    stepsize: Optional[float] = None
    arbitrary: Optional[AttributeRepr] = None

    def _to_orm(self, init):
        arbitrary = init.pop("arbitrary")
        obj = super()._to_orm(init)
        obj.arbitrary = arbitrary
        return obj


param_type = ParameterRepr.__fields__['type'].default
Parameter.type = param_type
Parameter.__annotations__['type'] = Literal[param_type]

reprs['Parameter'] = ParameterRepr


class ParameterRepr2(BaseRepr):
    _constructor = Parameter2
    type: Literal['Parameter2'] = 'Parameter2'
    name: str = Field(alias='label')
    value: float
    stepsize: Optional[float] = None
    _otherattr: Optional[int] = None
    arbitrary: Optional[AttributeRepr] = None

    def _to_orm(self, init):
        arbitrary = init.pop("arbitrary")
        obj = super()._to_orm(init)
        obj.arbitrary = arbitrary
        return obj


reprs["Parameter2"] = ParameterRepr2


class ComposedParameter:
    type: Literal['ComposedParameter'] = 'ComposedParameter'

    def __init__(self, name, add, func):
        self.name = name
        self.add = add
        self.func = FunctionHolder(func=func)


class FunctionHolder:
    def __init__(self, *, func=None, pickled_func=None):
        if func is not None:
            pickled_func = pickle.dumps(func).hex()
        elif pickled_func is not None:
            func = pickle.loads(bytes.fromhex(pickled_func))
        else:
            raise ValueError("Either func or pickled_func must be given")
        self.func = func
        self.pickled_func = pickled_func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"FuncHolder({self.func})"


class FunctionRepr(BaseRepr):
    _constructor = FunctionHolder
    pickled_func: str
    type: Literal['FunctionHolder'] = 'FunctionHolder'


reprs['FunctionHolder'] = FunctionRepr


class ComposedParameterRepr(BaseRepr):
    _constructor = ComposedParameter
    type: Literal['ComposedParameter'] = 'ComposedParameter'
    name: str
    add: bool
    func: FunctionRepr


reprs["ComposedParameter"] = ComposedParameterRepr

ParamsTypeDiscriminated = Annotated[
    Union[ParameterRepr, ComposedParameterRepr, ParameterRepr2], Field(discriminator='type')]


class PDFRepr(BaseRepr):
    _constructor = PDF
    type: Literal['PDF'] = 'PDF'
    name: str
    params: list[ParamsTypeDiscriminated]

    def _to_orm(self, init):
        params = init.pop('params')
        init['a'] = params[0]
        init['b'] = params[1]
        return super()._to_orm(init)

    @root_validator(pre=True)
    def validate_all_pre(cls, values):
        if cls.orm_mode():
            print(f"Validating all orm mode pre: {values}")
        else:
            print(f"Validating all repr mode pre: {values}")
        return values

    @root_validator
    def validate_all(cls, values):
        if cls.orm_mode():
            print(f"Validating all orm mode: {values}")
        else:
            print(f"Validating all repr mode: {values}")
        return values

    @validator('params')
    def validate_params(cls, v):
        print("Validating params, orm_mode = ", cls._context)
        if cls.orm_mode():
            print("orm, validate", v)
        else:
            print("serial, validate", v)
        print("Finished validating params, orm_mode = ", cls._context)
        return v

    @validator('params', pre=True)
    def validate_params_pre(cls, v):
        print(f"Validating params pre, context = {cls._context}, orm_mode = {cls.orm_mode()}")
        if cls.orm_mode():
            print("orm, validate pre", v)
        else:
            print("serial, validate pre", v)
        print("Finished validating params pre, orm_mode = ", cls._context)
        return v


reprs["PDF"] = PDFRepr
param1 = Parameter("param1", 1.0, 0.1)
param2 = Parameter2("param2", True)


def func(x):
    return x ** 2


param3 = ComposedParameter(name="param3", add=True, func=func)
print("objects")
print(param1)
print(param2)
print(param3)
param2_repr = ParameterRepr2.from_orm(param2)
param1_repr = ParameterRepr.from_orm(param1)
param3_repr = ComposedParameterRepr.from_orm(param3)

print("reprs")
print(param1_repr)
print(param2_repr)
print(param3_repr)

print("JSON")
json1 = param1_repr.json(exclude_none=True)
print(json1)
json2 = param2_repr.json(exclude_none=True)
print(json2)
json3 = param3_repr.json(exclude_none=True)

print("to repr from json")
param1_repr2 = ParameterRepr.parse_raw(json1)
param2_repr2 = ParameterRepr2.parse_raw(json2)
param3_repr2 = ComposedParameterRepr.parse_raw(json3)
print(param1_repr2)
print(param2_repr2)
print(param3_repr2)

print("to orm from repr")
param1_orm = param1_repr2.to_orm()
param2_orm = param2_repr2.to_orm()
param3_orm = param3_repr2.to_orm()
print(param1_orm)
print(param2_orm)
print(param3_orm)

print("=" * 20 + "PDF" + "=" * 20)
pdf = PDF("pdf", param1, param2)
pdf_repr = PDFRepr.from_orm(pdf)
print(pdf_repr)
jsonpdf = pdf_repr.json(exclude_none=True, by_alias=True)
print("JSON")
print(jsonpdf)
print("from json")
pdf_repr2 = PDFRepr.parse_raw(jsonpdf)
print(pdf_repr2)
print("to orm")
pdf_orm = pdf_repr2.to_orm()
print(pdf_orm)
print(f"Param a type: {type(pdf_orm.params[0])}")
print(f"Param b type: {type(pdf_orm.params[1])}")
