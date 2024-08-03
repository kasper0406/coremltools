from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Placeholder, Program, types
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.input_types import TensorType

from jax._src.lib.mlir import ir
from jaxlib.mlir.dialects.func import FuncOp
from jaxlib.mlir.dialects.stablehlo import AddOp, ReturnOp

class StableHloConverter:

    def __init__(self, opset_version: bool = None):
        self.opset_version = _target(opset_version) if opset_version is not None else None

        self.OPS = {
            # "func.func": self.build_func,

            "stablehlo.add": self.op_add,
            "func.return": self.op_return,
        }

        self.prog = mil.Program()

    def convert(self, module: ir.Module) -> Program:
        logger.info("Converting graph.")

        for func in module.body:
            # self.OPS[op.OPERATION_NAME](op)
            self.build_func(func)

        return self.prog

    def build_func(self, hlo_func: FuncOp):
        context = {} # Map from results to created variables

        func_inputs = {}
        for arg in hlo_func.arguments:
            func_inputs[arg.get_name()] = mb.placeholder(
                arg.type.shape, dtype=self.__get_dtype(arg.type.element_type), allow_rank0_input=True
            )

        with Function(func_inputs, opset_version=self.opset_version) as ssa_func:
            for name in func_inputs.keys():
                context[name] = ssa_func.inputs[name]

            ssa_func.set_outputs(self.process_block(context, hlo_func.body.blocks[0]))
            self.prog.add_function(str(hlo_func.name), ssa_func)

    def process_block(self, context, block: ir.Block):
        outputs = None
        for op in block:
            # Convention: Only the "return" op is returning from its building function
            # TODO: Check that "return" is always the last node!
            ret = self.OPS[op.OPERATION_NAME](context, op)
            if ret is not None:
                if outputs is not None:
                    raise ValueError("More than 1 return op in block!")
                outputs = ret
        return outputs
    
    def op_add(self, context, op: AddOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = mb.add(x=lhs, y=rhs)
        context[op.result.get_name()] = cml_op

    def op_return(self, context, op: ReturnOp):
        return [ context[result.get_name()] for result in op.operands ]

    def __get_dtype(self, element_type):
        if isinstance(element_type, ir.IntegerType):
            return types.int32
        raise ValueError(f"Unsupported type {element_type}")

    def __tensor_type(self, name: str, type: ir.RankedTensorType) -> TensorType:
        return TensorType(name=name, shape=type.shape, dtype=self.__get_dtype(type.element_type))
