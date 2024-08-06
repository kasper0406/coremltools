import jax
import jax.export
import jax.numpy as jnp
from jax.experimental import export
from jax._src.lib.mlir import ir
from jax._src.interpreters import mlir as jax_mlir
import numpy as np

from coremltools.converters.mil.frontend.stablehlo.load import load
import coremltools as ct
from coremltools.converters.mil.testing_utils import compare_backend

def test_addition():
    def plus(x, y):
        return jnp.add(x, y)

    run_and_compare(plus, (jnp.float32(1), jnp.float32(1)))
    run_and_compare(plus, (jnp.zeros((2, 2, 2)), jnp.zeros((2, 2, 2))))

def jax_export(jax_func, input_spec):
    input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in input_spec]
    jax_exported = export.export(jax.jit(jax_func))(*input_shapes)
    return jax_exported

def generate_random_from_shape(input_spec, key=jax.random.PRNGKey):
    shape = input_spec.shape
    dtype = input_spec.dtype
    output = jax.random.uniform(key=key, shape=shape, dtype=dtype, minval=-10, maxval=10)
    return output

def run_and_compare(jax_func, input_spec):
    jax_func = jax.jit(jax_func)
    exported = jax_export(jax_func, input_spec)
    context = jax_mlir.make_ir_context()
    hlo_module = ir.Module.parse(exported.mlir_module(), context=context)
    # print(f"HLO module: {hlo_module}")

    cml_model = ct.convert(hlo_module)

    cml_input_key_values = {}
    jax_input_values = []
    key = jax.random.PRNGKey(0)
    for input_name, input_shape in zip(cml_model.input_description, exported.in_avals):
        key, value_key = jax.random.split(key, num=2)
        input_value = generate_random_from_shape(input_shape, value_key)
        cml_input_key_values[input_name] = input_value
        jax_input_values.append(input_value)

    expected_output = jax_func(*jax_input_values)
    
    # TODO(knielsen): Is there a nicer way of doing this?
    if not isinstance(expected_output, (list, tuple)):
        expected_output = (expected_output, )

    cml_expected_outputs = {}
    for output_name, output_value in zip(cml_model.output_description, expected_output):
        cml_expected_outputs[output_name] = np.asarray(output_value)

    compare_backend(cml_model, cml_input_key_values, cml_expected_outputs)
