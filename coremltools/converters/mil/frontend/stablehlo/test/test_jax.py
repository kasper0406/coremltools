import jax
import jax.numpy as jnp
from jax.experimental import export
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir

from coremltools.converters.mil.frontend.stablehlo.load import load

def test_simple():
    def plus(x,y):
        return jnp.add(x,y)

    inputs = (jnp.int32(1), jnp.int32(1),)
    input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]
    stablehlo_add = export.export(plus)(*input_shapes).mlir_module()

    with jax_mlir.make_ir_context():
        stablehlo_module = ir.Module.parse(stablehlo_add)
        print(f"Module: {stablehlo_module}")

        prog = load(stablehlo_module)
        print(f"Generated prog: {prog}")
