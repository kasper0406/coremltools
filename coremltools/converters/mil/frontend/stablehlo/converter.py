from coremltools import _logger as logger
from coremltools.converters.mil import mil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Function, Program, types
from coremltools.converters.mil._deployment_compatibility import AvailableTarget as _target
from coremltools.converters.mil.input_types import TensorType

from jaxlib.mlir import ir
from jaxlib.mlir.dialects.func import FuncOp, CallOp, ReturnOp as FuncReturnOp
from jaxlib.mlir.dialects.stablehlo import AddOp, SubtractOp, MulOp, DivOp, NegOp, ExpOp, ConstantOp, DotGeneralOp, ReshapeOp, BroadcastInDimOp, WhileOp, CompareOp, ConvertOp, SelectOp, DynamicSliceOp, ReturnOp, ConvolutionOp, MaxOp, RsqrtOp

import numpy as np

from typing import Dict, List
import inspect
from dataclasses import dataclass
import re

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

@dataclass
class ConvDimensions:
    in_dim: List
    weights_dim: List
    out_dim: List


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
    def op_call(self, context: TranscriptionContext, op: CallOp):
        # We can not do function calls in MIL, so we have to inline the function

        # Get the argument mapping prior to entering the function context
        context_args = []

        for arg in op.operands:
            context_args.append(context[arg.get_name()])

        func_name = op.callee.value
        hlo_func = self.func_index[op.callee.value]
        params = hlo_func.arguments
        outputs = self.__invoke_hlo_function(context, func_name, params, hlo_func.body, context_args)

        # Configure return value
        for result, output in zip(op.results, outputs):
            context.add_variable(result.get_name(), output)

    @register_stablehlo_op
    def op_return(self, context: TranscriptionContext, op: ReturnOp):
        return [ context[result.get_name()] for result in op.operands ]

    @register_stablehlo_op
    def op_func_return(self, context: TranscriptionContext, op: FuncReturnOp):
        # The HLO / MLIR types for function return ops seem to be both in use
        # The behaviour and fields of the two types should be similar, so we
        # simply delegate to the HLO version
        return self.op_return(context, op)

    @register_stablehlo_op
    def op_add(self, context: TranscriptionContext, op: AddOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = mb.add(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_subtract(self, context: TranscriptionContext, op: SubtractOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = mb.sub(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_mul(self, context: TranscriptionContext, op: MulOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = mb.mul(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_div(self, context: TranscriptionContext, op: DivOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]

        # From HLO constraints we know the base-types should line up
        lhs_type = self.__resolve_type(lhs)
        rhs_type = self.__resolve_type(rhs)
        if lhs_type != rhs_type:
            raise ValueError(f"Division not supported for different types. lhs type: {lhs_type}, rhs type: {rhs_type}")
        if types.is_complex(lhs_type):
            raise ValueError("Complex numbers are not supported in MIL")

        if types.is_float(lhs_type):
            cml_op = mb.real_div(x=lhs, y=rhs)
        elif types.is_int(lhs_type):
            cml_op = mb.floor_div(x=lhs, y=rhs)
        else:
            raise ValueError(f"Unknown dtype {lhs_type}")

        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_neg(self, context: TranscriptionContext, op: NegOp):
        # TODO(knielsen): Consider unsigned and more exotic types
        operand = context[op.operand.get_name()]
        minus_one = np.array([-1], dtype=types.nptype_from_builtin(operand.dtype))
        cml_op = mb.mul(x=minus_one, y=operand)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_exp(self, context: TranscriptionContext, op: ExpOp):
        operand = context[op.operand.get_name()]
        cml_op = mb.exp(x=operand)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_constant(self, context: TranscriptionContext, op: ConstantOp):
        constant = np.array(op.value)
        context.add_variable(op.result.get_name(), constant)

    @register_stablehlo_op
    def op_dot_general(self, context: TranscriptionContext, op: DotGeneralOp):
        lhs_rank = len(op.lhs.type.shape)
        rhs_rank = len(op.rhs.type.shape)
        dims = self.__hacky_parse_attribute(op.dot_dimension_numbers)
        contract_lhs = [ int(d.strip()) for d in dims["lhs_contracting_dimensions"].strip("[]").split(",") ]
        contract_rhs = [ int(d.strip()) for d in dims["rhs_contracting_dimensions"].strip("[]").split(",") ]

        if len(contract_lhs) == 1 and contract_lhs[0] == lhs_rank - 1 and len(contract_rhs) == 1 and contract_rhs[0] == rhs_rank - 2:
            # This special case corresponds to CoreML matmul
            lhs = context[op.lhs.get_name()]
            rhs = context[op.rhs.get_name()]
            matmul_res = mb.matmul(x=lhs, y=rhs)
            context.add_variable(op.result.get_name(), matmul_res)
            return
        
        raise ValueError(f"The specified DotGeneral operation is not supported: {op}")

    @register_stablehlo_op
    def op_reshape(self, context: TranscriptionContext, op: ReshapeOp):
        x = context[op.operand.get_name()]
        new_shape = op.result.type.shape
        reshape_res = mb.reshape(x=x, shape=new_shape)
        context.add_variable(op.result.get_name(), reshape_res)

    @register_stablehlo_op
    def op_broadcast_in_dim(self, context: TranscriptionContext, op: BroadcastInDimOp):
        # TODO(knielsen): Consider if this is actually correct!
        # CoreML seems to auto-broadcast along the lines of numpy. Therefore this
        # explicit broadcasting op is not necessary.
        x = context[op.operand.get_name()]
        context.add_variable(op.result.get_name(), x)

    @register_stablehlo_op
    def op_while(self, context: TranscriptionContext, op: WhileOp):
        def cond(*loop_args):
            params = [ param for param in op.cond.blocks[0].arguments ]
            outputs = self.__invoke_hlo_function(context, "while_cond", params, op.cond, loop_args)
            if len(outputs) != 1:
                raise ValueError("The output of while_cond should always be a single boolean!")
            # TODO(knielsen): Add a check that the output is in fact a single boolean value

            return outputs[0]
        
        def body(*body_args):
            params = [ param for param in op.body.blocks[0].arguments ]
            return self.__invoke_hlo_function(context, "while_body", params, op.body, body_args)

        loop_vars = [ context[arg.get_name()] for arg in op.operands ]
        while_results = mb.while_loop(_cond=cond, _body=body, loop_vars=loop_vars)

        for result_var, while_result in zip(op.results, while_results):
            context.add_variable(result_var.get_name(), while_result)

    @register_stablehlo_op
    def op_compare(self, context: TranscriptionContext, op: CompareOp):
        comparison_direction = self.__hacky_parse_attribute(op.comparison_direction)["comparison_direction"]
        cml_op_builder = {
            "EQ": mb.equal,
            "NE": mb.not_equal,
            "GE": mb.greater_equal,
            "GT": mb.greater,
            "LE": mb.less_equal,
            "LT": mb.less,
        }[comparison_direction]

        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_op = cml_op_builder(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_convert(self, context: TranscriptionContext, op: ConvertOp):
        x = context[op.operand.get_name()]
        new_dtype = self.__get_dtype(op.result.type.element_type)
        cml_op = mb.cast(x=x, dtype=self.__dtype_str(new_dtype))
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_select(self, context: TranscriptionContext, op: SelectOp):
        cond = context[op.pred.get_name()]
        a = context[op.on_true.get_name()]
        b = context[op.on_false.get_name()]
        cml_op = mb.select(cond=cond, a=a, b=b)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_dynamic_slice(self, context: TranscriptionContext, op: DynamicSliceOp):
        x = context[op.operand.get_name()]

        # The HLO DynamicSliceOp gives the start indices as seperate 0-dimensional integer variables
        # We need to convert them to a tensor to be compatible with mb.slice_by_size
        start_idx_variables = [ context[i.get_name()] for i in op.start_indices ]
        begin = mb.concat(values=start_idx_variables, axis=0)

        # The slice sizes in HLO are given by a signed integer with 64 bits
        # This is not supported by MIL, so we convert it to a MIL int32 type
        # TODO(knielsen): Overflow check?
        sizes = np.array(op.slice_sizes, dtype=np.int32)

        cml_op = mb.slice_by_size(x=x, begin=begin, size=sizes)
        context.add_variable(op.result.get_name(), cml_op)

    @register_stablehlo_op
    def op_convolution(self, context: TranscriptionContext, op: ConvolutionOp):
        # TODO(knielsen): Support additional dimension specifications
        dim_spec = self.__hacky_parse_conv_dimensions(op.dimension_numbers)
        if dim_spec.in_dim[0] != "b" or dim_spec.out_dim[0] != "b":
            raise ValueError(f"Only the first dimension is currently supported for batch dimension. Got {dim_spec}")
        if dim_spec.in_dim[1] != "0" or dim_spec.out_dim[1] != "0" or dim_spec.weights_dim[0] != "0":
            raise ValueError(f"This must currently be 0. Not totally sure what it exactly means. Got {dim_spec}")
        if dim_spec.weights_dim[-1] != "o":
            raise ValueError(f"The output channels dim must be the last weight dimension. Got {dim_spec}")
        if dim_spec.weights_dim[-2] != "i":
            raise ValueError(f"The input channels dim must be the second last weight dimension. Got {dim_spec}")

        if op.batch_group_count.value != 1:
            raise ValueError(f"Only a batch group count of 1 is supported. Got {op.batch_group_count.value}")

        # The op.lhs has dimension [batch, d_in*, channels]
        # MIL expects it on the form [batch, channels, d_in*]
        x = context[op.lhs.get_name()] # The inputs comes from vars
        perm = list(range(x.rank))
        # Move the second axis to the end
        perm.append(perm.pop(1))
        x = mb.transpose(x=x, perm=perm)

        strides = None
        if op.window_strides is not None:
            strides = np.array(op.window_strides, dtype=np.int32)

        kernel_dilation = None
        if op.rhs_dilation is not None:
            kernel_dilation = np.array(op.rhs_dilation, dtype=np.int32)

        groups = op.feature_group_count.value

        # Handle padding
        # TODO(knielsen): Consider moving splat/non-splat handling to some utility
        in_rank = x.rank - 2
        if op.padding is None:
            pad = np.zeros((2 * in_rank), dtype=np.int32)
        elif op.padding.is_splat:
            pad = op.padding.get_splat_value().value * np.ones((2 * in_rank), dtype=np.int32)
        else:
            # We need to reshape the array to a linear array to match MILs expectation
            pad = np.reshape(np.array(op.padding, dtype=np.int32), (2 * in_rank, ))

        # We switch the convolution to a transposed convolution if we have lhs_dilation
        conv_type = mb.conv
        if op.lhs_dilation is not None:
            if strides is not None:
                raise ValueError("For a conv with lhs dilation we expect the stride to be not set! Because convolution with input dilation d is equivalent to transposed convolution with stride d.")
            # Convolution with input dilation d is equivalent to transposed convolution with stride d
            strides = np.array(op.lhs_dilation, dtype=np.int32)
            conv_type = mb.conv_transpose

        # The MIL weights should be on form:
        #  - normal convolutions: [C_out, C_in / groups, Kernel*]
        #  - transposed convolutions: [C_in, C_out / groups, Kernel*]
        # HLO has the form [Kernel*, C_in / groups, C_out]
        weight = context[op.rhs.get_name()] # The weights are numpy arrays
        perm = []
        # Move the channel dims
        if conv_type == mb.conv:
            perm.append(len(weight.shape) - 1)
            perm.append(len(weight.shape) - 2)
        else:
            perm.append(len(weight.shape) - 2)
            perm.append(len(weight.shape) - 1)
        for i in range(len(weight.shape) - 2):
            # Kernel perms moved to after the channels
            perm.append(i)
        weight = mb.transpose(x=weight, perm=perm)

        cml_conv = conv_type(
            x=x,
            weight=weight,
            strides=strides,
            pad_type="custom",
            pad=pad,
            dilations=kernel_dilation,
            groups=groups,
        )

        # Re-arrange output dimensions to match expectation
        # MIL outputs on the form [batch, channels, d_in*]
        # In the HLO program we expect [batch, d_in*, channels]
        perm = list(range(x.rank))
        # Move the second axis to the end
        perm.append(perm.pop(1))
        cml_conv = mb.transpose(x=cml_conv, perm=perm)

        context.add_variable(op.result.get_name(), cml_conv)

    @register_stablehlo_op
    def op_max(self, context: TranscriptionContext, op: MaxOp):
        lhs = context[op.lhs.get_name()]
        rhs = context[op.rhs.get_name()]
        cml_res = mb.maximum(x=lhs, y=rhs)
        context.add_variable(op.result.get_name(), cml_res)

    @register_stablehlo_op
    def op_rsqrt(self, context: TranscriptionContext, op: RsqrtOp):
        x = context[op.operand.get_name()]
        mil_res = mb.rsqrt(x=x)
        context.add_variable(op.result.get_name(), mil_res)

    def __invoke_hlo_function(self, context: TranscriptionContext, func_name: str, hlo_params, hlo_func_body, cml_args):
        # Enter variable context for the function call
        context.push_function(func_name)

        # Setup arguments for the function
        for hlo_func_param, actual_arg in zip(hlo_params, cml_args):
            context.add_variable(hlo_func_param.get_name(), actual_arg)
        
        # Process the function
        if len(hlo_func_body.blocks) != 1:
            raise ValueError(f"Unsupported function with {len(hlo_func_body.blocks)} blocks")
        outputs = self.process_block(context, hlo_func_body.blocks[0])

        # Exit the function context
        context.pop_function()

        return outputs

    # TODO(knielsen): Figure out a way to get rid of this!!!
    def __hacky_parse_attribute(self, attr: ir.Attribute):
        attr_str = str(attr)
        start = attr_str.find("<")
        end = attr_str.rfind(">")
        if start == -1 or end == -1:
            return {}

        attr_str = attr_str[(start + 1):end]
    
        attributes = {}
        # Split the string by comma to separate each key-value pair
        for attribute in attr_str.split(','):
            if "=" in attribute:
                # Split by '=' to separate keys and values
                key, value = attribute.split('=')
            else:
                # Assume space seperated
                key, value = attribute.split(' ')
        
            # Remove extra spaces and strip brackets
            key = key.strip()
            value = value.strip()
            attributes[key] = value

        return attributes

    def __hacky_parse_conv_dimensions(self, attr):
        # Example form: #stablehlo.conv<[b, 0, f]x[0, i, o]->[b, 0, f]>
        s = str(attr)

        matches = re.findall(r'\[([^\[\]]+)\]', s)
        lists = [ match.split(', ') for match in matches ]
        return ConvDimensions(
            in_dim=lists[0],
            weights_dim=lists[1],
            out_dim=lists[2],
        )

    def __resolve_type(self, obj):
        if isinstance(obj, np.ndarray):
            return types.numpy_type_to_builtin_type(obj.dtype)
        return obj.dtype

    def __dtype_str(self, type):
        # TODO(knielsen): Add additional types
        return {
            types.int32: "int32",
            types.fp16: "fp16",
            types.fp32: "fp32",
        }[type]

    def __get_dtype(self, element_type):
        if isinstance(element_type, ir.IntegerType):
            # TODO(knielsen): Handle different kinds of integer types
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
