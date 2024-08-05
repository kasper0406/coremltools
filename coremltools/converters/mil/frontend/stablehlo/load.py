from coremltools.converters.mil.mil import Program
from coremltools.converters.mil.frontend.stablehlo.converter import StableHloConverter
from jax._src.lib.mlir import ir

def load(module: ir.Module, **kwargs) -> Program:
    converter = StableHloConverter()

    return converter.convert(module)
