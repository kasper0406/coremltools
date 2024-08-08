from flax import nnx
import jax
import jax.numpy as jnp
from jax._src.typing import Array
from jax._src import core
from jax._src import dtypes

from functools import partial
from typing import Any, Optional
import einops

KeyArray = Any
DTypeLikeFloat = Any
DTypeLikeComplex = Any
DTypeLikeInexact = Any  # DTypeLikeFloat | DTypeLikeComplex
RealNumeric = Any  # Scalar jnp array or float

def uniform(minval: RealNumeric = 0,
            maxval: RealNumeric = 1,
            dtype: DTypeLikeInexact = jnp.float_) -> nnx.Initializer:
  def init(key: KeyArray,
           shape: core.Shape,
           dtype: DTypeLikeInexact = dtype) -> Array:
    dtype = dtypes.canonicalize_dtype(dtype)
    return jax.random.uniform(key, shape, dtype, minval=minval, maxval=maxval)
  return init

def hacky_scan_wrapper(f, x, y):
    """
    Unfortunately is nnx is not too fond of nnx.scan when exporting
    """
    return jax.lax.scan(f, x, y)
    def wrapped_f(x, y):
        with jax.disable_jit(False):
            return f(x, y)

    with jax.disable_jit():
        return jax.lax.scan(wrapped_f, x, y, unroll=True)

class sLSTMCell(nnx.Module):
    cell_input_proj: nnx.Linear
    cell_state_proj: nnx.Linear

    input_proj: nnx.Linear
    input_state_proj: nnx.Linear
    
    forget_gate_proj: nnx.Linear
    forget_state_proj: nnx.Linear

    output_gate_proj: nnx.Linear
    output_state_proj: nnx.Linear

    if_conv: nnx.Conv | None = None

    def __init__(self, num_cells: int, rngs: nnx.Rngs, input_size: Optional[int] = None, apply_if_conv: bool = True):
        construct_x_proj = partial(nnx.Linear,
            in_features=input_size if input_size else num_cells,
            out_features=num_cells,
            use_bias=True,
            rngs=rngs,
        )
        construct_hidden_proj = partial(nnx.Linear,
            in_features=num_cells,
            out_features=num_cells,
            use_bias=False,
            kernel_init=nnx.initializers.orthogonal(),
            rngs=rngs,
        )

        self.cell_input_proj = construct_x_proj()
        self.cell_state_proj = construct_hidden_proj()

        self.input_proj = construct_x_proj(
            bias_init=nnx.initializers.normal(1e-2)
        )
        self.input_state_proj = construct_hidden_proj()

        self.forget_gate_proj = construct_x_proj(
            bias_init=uniform(3.0, 6.0)
        )
        self.forget_state_proj = construct_hidden_proj()

        self.output_gate_proj = construct_x_proj()
        self.output_state_proj = construct_hidden_proj()

        if apply_if_conv:
            self.if_conv = nnx.Conv(
                in_features=1,
                out_features=1,
                kernel_size=4,
                rngs=rngs
            )
    
    def __call__(self, carry, x):
        cell_state, hidden_state, normalizer_state, stabilizer_state = carry

        # print(f"Hidden state shape: {hidden_state.shape}")
        out = nnx.sigmoid(self.output_gate_proj(x) + self.output_state_proj(hidden_state))

        if_input = x
        if self.if_conv:
            if_input = self.if_conv(if_input[..., jnp.newaxis])
            if_input = jnp.squeeze(nnx.silu(if_input), axis=2)

        i_tilde = self.input_proj(if_input) + self.input_state_proj(hidden_state)
        # TODO: Consider trying a sigmoid forget activation as well!
        f_tilde = self.forget_gate_proj(if_input) + self.forget_state_proj(hidden_state)

        m = jnp.maximum(f_tilde + stabilizer_state, i_tilde) # Stabilizer state
        i = jnp.exp(i_tilde - m)
        f = jnp.exp(f_tilde + stabilizer_state - m)

        z = nnx.tanh(self.cell_input_proj(x) + self.cell_state_proj(hidden_state))
        c = f * cell_state + i * z # Cell state

        n = f * normalizer_state + i # Normalizer state
        h = out * c / n # Hidden state

        return (c, h, n, m), h

    @classmethod
    def init_carry(cls, batch_size: int, num_cells: int, rngs: nnx.Rngs):
        initializer = partial(nnx.initializers.zeros, shape=(batch_size, num_cells), dtype=jnp.float16)
        c = initializer(key=rngs())
        h = initializer(key=rngs())
        n = initializer(key=rngs())
        m = initializer(key=rngs())
        return c, h, n, m

class sLSTMBlock(nnx.Module):
    heads: sLSTMCell
    input_norm: nnx.BatchNorm
    output_norm: nnx.BatchNorm

    up_proj_1: nnx.Linear
    up_proj_2: nnx.Linear
    down_proj: nnx.Linear

    def __init__(self, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        @partial(nnx.vmap, axis_size=num_heads)
        def create_heads(rngs: nnx.Rngs):
            return sLSTMCell(hidden_size, rngs)
        self.heads = create_heads(rngs)

        self.input_norm = nnx.BatchNorm(hidden_size, rngs=rngs)
        self.output_norm = nnx.BatchNorm(hidden_size, rngs=rngs) # reduction_axes=..., feature_axes=...)

        intermediate_size = int(hidden_size * num_heads * 4.0 / 3.0)
        self.up_proj_1 = nnx.Linear(
            in_features=hidden_size * num_heads,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.up_proj_2 = nnx.Linear(
            in_features=hidden_size * num_heads,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            rngs=rngs,
        )

    def __call__(self, carry, x):
        out = self.input_norm(x)

        slstm_defs, slstm_states = nnx.split(self.heads)
        @partial(jax.vmap, in_axes=(0, 0, None))
        def call_slstm(slstm_state, carry, x):
            slstm = nnx.merge(slstm_defs, slstm_state)
            return slstm(carry, x)
        carry, out = call_slstm(slstm_states, carry, out)

        out = self.output_norm(out)
        out = einops.rearrange(out, "heads batch hidden -> batch (heads hidden)")

        out = self.up_proj_1(out) * nnx.gelu(self.up_proj_2(out))
        out = self.down_proj(out)

        out += x # Residual

        return carry, out

    @classmethod
    def init_carry(cls, batch_size: int, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        @partial(nnx.vmap, axis_size=num_heads)
        def create_carry(rngs: nnx.Rngs):
            return sLSTMCell.init_carry(batch_size, hidden_size, rngs)
        return create_carry(rngs)

class mLSTMCell(nnx.Module):
    query_proj: nnx.Linear
    key_proj: nnx.Linear
    value_proj: nnx.Linear
    qk_conv: nnx.Conv

    input_proj: nnx.Linear
    forget_proj: nnx.Linear
    output_proj: nnx.Linear

    learnable_skip: nnx.Linear

    hidden_size: int

    def __init__(self, hidden_size: int, rngs: nnx.Rngs, input_size: Optional[int] = None):
        self.hidden_size = hidden_size

        construct_qkv_proj = partial(nnx.Linear,
            in_features=input_size if input_size else hidden_size,
            out_features=hidden_size,
            use_bias=True,
            rngs=rngs,
        )
        construct_x_proj = partial(nnx.Linear,
            in_features=input_size if input_size else hidden_size,
            out_features=1,
            use_bias=True,
            rngs=rngs,
        )

        self.query_proj = construct_qkv_proj()
        self.key_proj = construct_qkv_proj()
        self.value_proj = construct_qkv_proj()
        self.qk_conv = nnx.Conv(
            in_features=1,
            out_features=1,
            kernel_size=4,
            rngs=rngs
        )

        self.input_proj = construct_x_proj(
            bias_init=nnx.initializers.normal(1e-2)
        )

        self.forget_proj = construct_x_proj(
            bias_init=uniform(3.0, 6.0)
        )

        self.output_proj = construct_x_proj(out_features=self.hidden_size)

        self.learnable_skip = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            rngs=rngs,
        )

    def __call__(self, carry, x):
        cell_state, normalizer_state, stabilizer_state = carry
        # HACK: See init_carry for an explanation
        cell_state = einops.rearrange(cell_state, "(b h1 h2) -> b h1 h2", h1=self.hidden_size, h2=self.hidden_size)
        
        out = nnx.sigmoid(self.output_proj(x)) # + self.output_state_proj(hidden_state))

        i_tilde = jnp.squeeze(self.input_proj(x), axis=1)
        f_tilde = jnp.squeeze(self.forget_proj(x), axis=1)

        # print(f"i_tilde shape: {i_tilde.shape}")
        # print(f"f_tilde shape: {f_tilde.shape}")
        # print(f"stabilizer_state shape: {stabilizer_state.shape}")

        m = jnp.maximum(f_tilde + stabilizer_state, i_tilde) # Stabilizer state
        i = jnp.exp(i_tilde - m)
        f = jnp.exp(f_tilde + stabilizer_state - m)

        qk_input = self.qk_conv(x[..., jnp.newaxis])
        qk_input = jnp.squeeze(nnx.silu(qk_input), axis=2)

        key = self.key_proj(qk_input) / jnp.sqrt(self.hidden_size)
        query = self.query_proj(qk_input)
        value = self.value_proj(x)

        # print(f"f shape: {f.shape}")
        c = jnp.einsum("b,bij->bij", f, cell_state) + jnp.einsum("b,bi,bj->bij", i, value, key)
        n = jnp.einsum("b,bi->bi", f, normalizer_state) + jnp.einsum("b,bi->bi", i, key)

        scaler = jnp.abs(jnp.einsum("bi,bi->b", n, query))
        scaler = 1 / jnp.maximum(scaler, 1.0)
        h = out * jnp.einsum("bik,bk,b->bi", c, query, scaler)

        h += self.learnable_skip(qk_input)

        # HACK: See init_carry for an explanation
        c = einops.rearrange(c, "b h1 h2 -> (b h1 h2)")
        return (c, n, m), h

    @classmethod
    def init_carry(cls, batch_size: int, hidden_size: int, rngs: nnx.Rngs):
        c = nnx.initializers.zeros(shape=(batch_size, hidden_size, hidden_size), dtype=jnp.float16, key=rngs())
        # HACK: CoreML does not supprot tensors of rank > 5, so we squeeze the c carry to make it fit
        c = einops.rearrange(c, "b h1 h2 -> (b h1 h2)")

        n = nnx.initializers.zeros(shape=(batch_size, hidden_size), dtype=jnp.float16, key=rngs())
        m = nnx.initializers.zeros(shape=(batch_size,), dtype=jnp.float16, key=rngs())

        return c, n, m

class mLSTMBlock(nnx.Module):
    heads: mLSTMCell
    input_norm: nnx.BatchNorm
    output_norm: nnx.BatchNorm

    up_proj_1: nnx.Linear
    up_proj_2: nnx.Linear
    down_proj: nnx.Linear
    head_polling: nnx.Linear

    def __init__(self, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        @partial(nnx.vmap, axis_size=num_heads)
        def create_heads(rngs: nnx.Rngs):
            return mLSTMCell(2 * hidden_size, rngs)
        self.heads = create_heads(rngs)

        self.input_norm = nnx.BatchNorm(hidden_size, rngs=rngs)
        self.output_norm = nnx.BatchNorm(2 * hidden_size, rngs=rngs) # reduction_axes=..., feature_axes=...)

        intermediate_size = hidden_size * 2
        self.up_proj_1 = nnx.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.up_proj_2 = nnx.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            rngs=rngs,
        )
        self.down_proj = nnx.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            rngs=rngs,
        )
        self.head_polling = nnx.Linear(
            in_features=num_heads * intermediate_size,
            out_features=intermediate_size,
            rngs=rngs,
        )

    def __call__(self, carry, x):
        out = self.input_norm(x)

        triggers = nnx.silu(self.up_proj_2(out))
        mlstm_input = self.up_proj_1(out)

        mlstm_def, mlstm_layer_states = nnx.split(self.heads)
        @partial(jax.vmap, in_axes=(0, 0, None))
        def call_mlstm(mlstm_state, carry, x):
            mlstm = nnx.merge(mlstm_def, mlstm_state)
            return mlstm(carry, x)
        carry, mlstm_out = call_mlstm(mlstm_layer_states, carry, mlstm_input)
        mlstm_out = self.output_norm(mlstm_out)
        mlstm_out = einops.rearrange(mlstm_out, "heads batch hidden -> batch (heads hidden)")
        mlstm_out = self.head_polling(mlstm_out)

        out = mlstm_out * triggers
        out = self.down_proj(out)

        out += x # residual

        return carry, out

    @classmethod
    def init_carry(cls, batch_size: int, hidden_size: int, num_heads: int, rngs: nnx.Rngs):
        @partial(nnx.vmap, axis_size=num_heads)
        def create_carry(rngs: nnx.Rngs):
            return mLSTMCell.init_carry(batch_size, 2 * hidden_size, rngs)
        return create_carry(rngs)

class xLSTMModule(nnx.Module):
    num_mlstm: int
    num_slstm: int
    hidden_size: int
    num_heads: int

    mlstms: mLSTMBlock
    slstms: sLSTMBlock

    def __init__(self, hidden_size: int, num_heads: int, num_mlstm: int, num_slstm: int, rngs: nnx.Rngs):
        self.num_mlstm = num_mlstm
        self.num_slstm = num_slstm
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        @partial(nnx.vmap, axis_size=num_mlstm)
        def create_mlstms(rngs: nnx.Rngs):
            return mLSTMBlock(hidden_size, num_heads, rngs=rngs)
        self.mlstms = create_mlstms(rngs)

        @partial(nnx.vmap, axis_size=num_slstm)
        def create_slstms(rngs: nnx.Rngs):
            return sLSTMBlock(hidden_size, num_heads, rngs=rngs)
        self.slstms = create_slstms(rngs)

    def __call__(self, carry, x):
        mlstm_carry, slstm_carry = carry

        mlstm_def, mlstm_states = nnx.split(self.mlstms)
        slstm_def, slstm_states = nnx.split(self.slstms)

        def forward(x, spec, defs):
            state, carry = spec
            xlstm = nnx.merge(defs, state)
            carry, y = xlstm(carry, x)
            return y, carry

        out = x
        out, mlstm_carry = hacky_scan_wrapper(
            partial(forward, defs=mlstm_def),
            out,
            (mlstm_states, mlstm_carry))

        out, slstm_carry = hacky_scan_wrapper(
            partial(forward, defs=slstm_def),
            out,
            (slstm_states, slstm_carry))

        return (mlstm_carry, slstm_carry), out
    
    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        @partial(nnx.vmap, axis_size=self.num_mlstm)
        def create_mlstm_carry(rngs: nnx.Rngs):
            return mLSTMBlock.init_carry(batch_size, self.hidden_size, self.num_heads, rngs)
        
        @partial(nnx.vmap, axis_size=self.num_slstm)
        def create_slstm_carry(rngs: nnx.Rngs):
            return sLSTMBlock.init_carry(batch_size, self.hidden_size, self.num_heads, rngs)

        return (create_mlstm_carry(rngs), create_slstm_carry(rngs))

class xLSTM(nnx.Module):
    layers: xLSTMModule

    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, rngs: nnx.Rngs, mlstm_per_layer: int = 1, slstm_per_layer: int = 1):
        @partial(nnx.vmap, axis_size=num_layers)
        def create_layers(rngs: nnx.Rngs):
            return xLSTMModule(hidden_size, num_heads, mlstm_per_layer, slstm_per_layer, rngs)
        self.layers = create_layers(rngs)

    def __call__(self, carry, x):
        layer_def, layers_state = nnx.split(self.layers)

        def forward(x, spec):
            xlstm_state, carry = spec
            xlstm = nnx.merge(layer_def, xlstm_state)
            carry, y = xlstm(carry, x)
            return y, carry

        out, carry = hacky_scan_wrapper(forward, x, (layers_state, carry))
        return carry, out

    def init_carry(self, batch_size: int, rngs: nnx.Rngs):
        @partial(nnx.vmap)
        def create_carry(xlstm, rngs):
            return xlstm.init_carry(batch_size, rngs)
        return create_carry(self.layers, rngs)
