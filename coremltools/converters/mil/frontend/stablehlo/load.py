from coremltools.converters.mil.mil import Program
from coremltools.converters.mil.frontend.stablehlo.converter import StableHloConverter
from jax._src.lib.mlir import ir

def load(module: ir.Module, specification_version: int, **kwargs) -> Program:
    converter = StableHloConverter(specification_version)

    return converter.convert(module)
