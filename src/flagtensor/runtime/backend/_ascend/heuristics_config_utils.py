# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tile heuristics for the Ascend backend.

The ``elementwise_{unary,binary,trinary}`` entries are shared by every Ascend
pointwise kernel, including the ones that still launch one program per tile
(e.g. ``conj``, the indexed binary kernel, the trinary kernels).  Those keep
the historical heuristics.

Ascend kernels that use the *persistent* launch (one program per vector core,
each program walking a strided range of tiles) get dedicated entries: they can
afford much larger tiles because the tile is not tied to the grid size, and
measurements show that 512-element transactions are far too small to saturate
HBM.

Measured on Ascend 910 across the benchmark shapes:

* The per-block launch overhead is significant, so many tiny tiles are slower
  than a few large ones, even for small tensors.
* Up to ~8K elements the operation is dominated by kernel launch latency and a
  single large tile (one program) is fastest.
* Above that the op is HBM-bandwidth bound; 8K-element transactions are the
  sweet spot for fp32 with two buffers.

The persistent tile is therefore the smallest power of two >= ``n_elements``,
clamped to [``ASCEND_MIN_TILE``, ``max_tile``].
"""

#: Tile used for tensors smaller than this.
ASCEND_MIN_TILE = 1024
#: Largest tile for unary kernels with many live fp32 temporaries
#: (asin/acos/asinh).  Their polynomial evaluation needs ~12 UB buffers; at
#: 8192 elements the fp32 pipeline overflows the unified buffer and
#: ``bishengir-compile`` fails.
ASCEND_MAX_TILE_UNARY_HEAVY = 4096
#: Largest tile for a 2-operand (in/out) fp32 kernel.
ASCEND_MAX_TILE_UNARY = 8192
#: Largest tile for unary kernels with a single cheap arithmetic/rounding op
#: (exp/sqrt/floor/ceil).  Only two or three UB buffers are live, so 16384
#: elements still fit and the smaller loop-trip count is measurably faster.
ASCEND_MAX_TILE_UNARY_WIDE = 16384
#: The wider tile is only used from this length on.  The persistent grid walks
#: ``ceil(n / tile)`` tiles over ``vector_core_num()`` (48) programs, so the
#: core-to-core imbalance is ``ceil(tiles / 48) / (tiles / 48)``.  Below ~4M
#: elements the 16384 tile leaves too few tiles and the imbalance (up to +50%
#: at 2M) costs more than the saved loop iterations.
ASCEND_WIDE_TILE_MIN_ELEMENTS = 4 * 1024 * 1024
#: Number of programs the heavy kernels aim for.  Their polynomial needs
#: ~15 vector ops per element, so below full occupancy the kernel is latency
#: bound and shrinking the tile (more programs) is faster than one big tile.
ASCEND_HEAVY_TILE_TARGET_PROGRAMS = 16
#: Largest tile for the 3-operand (2 in + out) persistent kernel.  ``min``/
#: ``max`` lower to a single ``maximumf``/``minimumf`` instruction with
#: ``propagate_nan=ALL`` and therefore need no extra temporary, so both the
#: 2-byte and the fp32 kernels fit 8192 elements (bf16 fails to compile at
#: 16384).  8192 is also the measured optimum for fp32.
ASCEND_MAX_TILE_BINARY = 8192
ASCEND_MAX_TILE_BINARY_FP32 = 8192


def simple_elementwise_blocksize_heur(args):
    n_elements = args['n_elements']
    if n_elements <= 512:
        return 256
    if n_elements <= 4096:
        return 512
    return 1024


def simple_elementwise_blocks_per_program_heur(args):
    n_elements = args['n_elements']
    if n_elements <= 4096:
        return 1
    if n_elements <= (1 << 20):
        return 2
    return 4


def _next_power_of_two(value):
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _persistent_tile(n_elements, max_tile):
    tile = _next_power_of_two(int(n_elements))
    return max(ASCEND_MIN_TILE, min(max_tile, tile))


def persistent_unary_blocksize_heur(args):
    return _persistent_tile(args['n_elements'], ASCEND_MAX_TILE_UNARY)


def _binary_max_tile(args):
    """Tile cap for the 3-operand persistent kernel, based on input width."""
    try:
        element_size = args['x_ptr'].element_size()
    except Exception:
        return ASCEND_MAX_TILE_BINARY_FP32
    if element_size >= 4:
        return ASCEND_MAX_TILE_BINARY_FP32
    return ASCEND_MAX_TILE_BINARY


def persistent_binary_blocksize_heur(args):
    return _persistent_tile(args['n_elements'], _binary_max_tile(args))


def persistent_unary_wide_blocksize_heur(args):
    n_elements = int(args['n_elements'])
    cap = (
        ASCEND_MAX_TILE_UNARY_WIDE
        if n_elements >= ASCEND_WIDE_TILE_MIN_ELEMENTS
        else ASCEND_MAX_TILE_UNARY
    )
    return _persistent_tile(n_elements, cap)


def persistent_unary_wide_even_heur(args):
    n_elements = int(args['n_elements'])
    tile = persistent_unary_wide_blocksize_heur(args)
    return 1 if n_elements % tile == 0 else 0


def persistent_unary_heavy_blocksize_heur(args):
    # Aim for ASCEND_HEAVY_TILE_TARGET_PROGRAMS programs: at small sizes the
    # polynomial latency is hidden by spreading the tensor over more cores,
    # at large sizes the tile is capped by the unified buffer instead.
    n_elements = int(args['n_elements'])
    tile = _next_power_of_two(max(1, n_elements // ASCEND_HEAVY_TILE_TARGET_PROGRAMS))
    return max(ASCEND_MIN_TILE, min(ASCEND_MAX_TILE_UNARY_HEAVY, tile))


def persistent_unary_heavy_even_heur(args):
    n_elements = int(args['n_elements'])
    tile = persistent_unary_heavy_blocksize_heur(args)
    return 1 if n_elements % tile == 0 else 0


def persistent_unary_even_heur(args):
    n_elements = int(args['n_elements'])
    tile = _persistent_tile(n_elements, ASCEND_MAX_TILE_UNARY)
    return 1 if n_elements % tile == 0 else 0


def persistent_binary_even_heur(args):
    n_elements = int(args['n_elements'])
    tile = _persistent_tile(n_elements, _binary_max_tile(args))
    return 1 if n_elements % tile == 0 else 0


def persistent_blocks_per_program_heur(args):
    # The persistent loop already walks a strided range of tiles, so one tile
    # per iteration is enough; multiplying the tile only inflates UB usage.
    return 1


HEURISTICS_CONFIGS = {
    'elementwise_unary': {
        'BLOCK_SIZE': simple_elementwise_blocksize_heur,
        'BLOCKS_PER_PROGRAM': simple_elementwise_blocks_per_program_heur,
    },
    'elementwise_binary': {
        'BLOCK_SIZE': simple_elementwise_blocksize_heur,
        'BLOCKS_PER_PROGRAM': simple_elementwise_blocks_per_program_heur,
    },
    'elementwise_trinary': {
        'BLOCK_SIZE': simple_elementwise_blocksize_heur,
        'BLOCKS_PER_PROGRAM': simple_elementwise_blocks_per_program_heur,
    },
    # Used by the Ascend-only persistent kernels.
    'elementwise_unary_persistent': {
        'BLOCK_SIZE': persistent_unary_blocksize_heur,
        'BLOCKS_PER_PROGRAM': persistent_blocks_per_program_heur,
        'EVEN': persistent_unary_even_heur,
    },
    # Unary kernels whose per-element work needs more UB temporaries.
    'elementwise_unary_persistent_heavy': {
        'BLOCK_SIZE': persistent_unary_heavy_blocksize_heur,
        'BLOCKS_PER_PROGRAM': persistent_blocks_per_program_heur,
        'EVEN': persistent_unary_heavy_even_heur,
    },
    # Unary kernels with very few live buffers (exp/sqrt/floor/ceil); they can
    # use a 2x larger tile than the generic unary entry.
    'elementwise_unary_persistent_wide': {
        'BLOCK_SIZE': persistent_unary_wide_blocksize_heur,
        'BLOCKS_PER_PROGRAM': persistent_blocks_per_program_heur,
        'EVEN': persistent_unary_wide_even_heur,
    },
    'elementwise_binary_persistent': {
        'BLOCK_SIZE': persistent_binary_blocksize_heur,
        'BLOCKS_PER_PROGRAM': persistent_blocks_per_program_heur,
        'EVEN': persistent_binary_even_heur,
    },
}
