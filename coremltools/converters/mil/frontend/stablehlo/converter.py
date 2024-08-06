from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.input_types import TensorType

from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects.func import FuncOp, CallOp, ReturnOp
from jaxlib.mlir.dialects.stablehlo import AddOp

from typing import Dict
import inspect

class TranscriptionContext:
    def __init__(self):
        self._path = []
        self.seen_paths = set()
        self.variables = {} # Nest map: path -> variable -> mil var

    def push_function(self, name: str):
        counter = 0
        ctx_name = name
        while True:
            new_path = self._path + [ctx_name]
            if "/".join(new_path) in self.seen_paths:
                # Ensure that the new context name is in fact unique
                # A collision can happen if the same function is called twice
                ctx_name = f"{name}_{counter}"
            else:
                self._path.append(ctx_name)
                self.seen_paths.add(self.path())
                return ctx_name
    
    def pop_function(self):
        self.variables.pop(self.path())
        self._path.pop()
    
    def add_variable(self, name: str, mil_var):
        path = self.path()
        if path not in self.variables:
            self.variables[path] = {}
        
        if name in self.variables[path]:
            raise ValueError(f"Variable {name} is already defined in path {path}")
        self.variables[path][name] = mil_var

    def __getitem__(self, name: str):
        path = self.path()
        ctx = self.variables[path]
        if name not in ctx:
            raise ValueError(f"Variable with name {name} is not defined in path {path}")
        return ctx[name]

    def path(self) -> str:
        return "/".join(self._path)

def register_stablehlo_op(func):
    # Check the signature
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Exclude 'self' from the parameters
    params = params[1:]

    error_msg = "HLO op implementations should take parameters of exactly (context: TranscriptionContext, op: <HLO_OP_TYPE>)"
    if len(params) != 2:
        raise TypeError(error_msg)
    
    if not issubclass(params[0].annotation, TranscriptionContext):
        raise TypeError(error_msg)

    # We identify the function by the type of operation it implements
    func._implements_hlo_op = params[1].annotation
    return func

class StableHloOpsRegistry(type):
    def __init__(cls, name, bases, clsdict):
        super().__init__(name, bases, clsdict)

        cls._stablehlo_ops_registry = {}
        for name, method in clsdict.items():
            op_type = getattr(method, '_implements_hlo_op', False)
            if callable(method) and op_type:
                if op_type in cls._stablehlo_ops_registry:
                    raise TypeError(f"StableHLO op {op_type} has been registered more than once!")
                cls._stablehlo_ops_registry[op_type] = method
    
    def _dispatch_op(cls, self, context: TranscriptionContext, op):
        if type(op) not in self._stablehlo_ops_registry:
            raise TypeError(f"The StableHLO op {type(op)} has not been implemented!")
        
        op_method = self._stablehlo_ops_registry[type(op)]
        return op_method(self, context, op)

    def __call__(cls, *args, **kwargs):
        # Register the dispatch_op method
        instance = super().__call__(*args, **kwargs)
        setattr(instance, 'dispatch_op', cls._dispatch_op)
        return instance

class StableHloConverter(metaclass=StableHloOpsRegistry):

    func_indx: Dict[str, FuncOp]

    def __init__(self, opset_version: bool = None):
        self.opset_version = _target(opset_version) if opset_version is not None else None
        self.prog = mil.Program()
        self.func_index = {}

    def convert(self, module: ir.Module) -> Program:
        logger.info("Converting graph.")

        # Build function index to resolve/inline HLO function calls
        for func in module.body:
            self.func_index[func.name.value] = func

        for func in module.body:
            if "public" == func.visibility.value:
                self.build_func(func)

        return self.prog

    def build_func(self, hlo_func: FuncOp):
        context = TranscriptionContext() # Map from results to created variables

        func_inputs = {}
        for arg in hlo_func.arguments:
            shape = arg.type.shape
            if shape == []:
                shape = [1]

            func_inputs[arg.get_name()] = mb.placeholder(
                shape=shape, dtype=self.__get_dtype(arg.type.element_type)
            )

        with Function(func_inputs, opset_version=self.opset_version) as ssa_func:
            for name in func_inputs.keys():
                context.add_variable(name, ssa_func.inputs[name])

            ssa_func.set_outputs(self.process_block(context, hlo_func.body.blocks[0]))
            self.prog.add_function(hlo_func.name.value, ssa_func)

    def process_block(self, context: TranscriptionContext, block: ir.Block):
        outputs = None
        for op in block:
            # Convention: Only the "return" op is returning from its building function
            # TODO: Check that "return" is always the last node!
            ret = self.dispatch_op(self, context, op)
            if ret is not None:
                if outputs is not None:
                    raise ValueError("More than 1 return op in block!")
                outputs = ret
        return outputs
    
    @register_stablehlo_op
    def op_add(self, context: TranscriptionContext, op: AddOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = mb.add(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_call(self, context: TranscriptionContext, op: CallOp):
        # We can not do function calls in MIL, so we have to inline the function

        # Get the argument mapping prior to entering the function context
        context_args = []
        for arg in op.operands:
            context_args.append(context[arg.get_name()])

        ctx_name = context.push_function(op.callee.value)

        # Set up parameter to argument mapping
        hlo_func = self.func_index[ctx_name]
        # Setup arguments for the function
        for hlo_func_param, actual_arg in zip(hlo_func.arguments, context_args):
            context.add_variable(hlo_func_param.get_name(), actual_arg)

        # Process the function
        outputs = self.process_block(context, hlo_func.body.blocks[0])

        context.pop_function()

        # Configure return value
        for result, output in zip(op.results, outputs):
            context.add_variable(result.get_name(), output)

    @register_stablehlo_op
    def op_return(self, context: TranscriptionContext, op: ReturnOp):
        return [ context[result.get_name()] for result in op.operands ]

    def __get_dtype(self, element_type):
        if isinstance(element_type, ir.IntegerType):
            return types.int32
        if isinstance(element_type, ir.F16Type):
            return types.fp16
        if isinstance(element_type, ir.F32Type):
            return types.fp32
        raise ValueError(f"Unsupported type {element_type}")

    def __tensor_type(self, name: str, type: ir.RankedTensorType) -> TensorType:
        shape = type.shape
        if shape == []:
            shape = [1]

        return TensorType(name=name, shape=shape, dtype=self.__get_dtype(type.element_type))
