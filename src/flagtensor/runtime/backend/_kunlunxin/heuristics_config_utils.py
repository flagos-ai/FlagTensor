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


# ---------------------------------------------------------------------------
# Memory-bound pointwise programs
#
# On XPU the *host* cost of a launch grows with the program count (roughly
# 0.03us per program plus a ~5us floor) while each program streams its tile
# at a fixed bandwidth.  A 4096-element tile therefore caps a large copy at
# ~300 GB/s, whereas a 16384-element tile reaches ~1 TB/s and still hides
# the launch charge.  Measured on P800 (bf16, 8 MiB):
#
#   tile   4096    8192   16384
#   8 MiB  114us    60us    34us
#
# Small inputs stay on small tiles: they are already at the ~5us launch
# floor and large tiles there only reduce occupancy.
_UNARY_TILE_LARGE = 16384       # BLOCK_SIZE * BLOCKS_PER_PROGRAM
_UNARY_TILE_LARGE_FROM = 1 << 18  # 256K elements


def unary_blocksize_heur(args):
    n_elements = args['n_elements']
    if n_elements <= 512:
        return 256
    if n_elements <= 4096:
        return 512
    if n_elements < _UNARY_TILE_LARGE_FROM:
        return 1024
    return 4096


def unary_blocks_per_program_heur(args):
    n_elements = args['n_elements']
    if n_elements <= 4096:
        return 1
    if n_elements < _UNARY_TILE_LARGE_FROM:
        return 4
    return _UNARY_TILE_LARGE // 4096


# The conj kernel walks the *interleaved* real/imag buffer, so its tile is
# twice the logical element count.  ``tl.arange`` is capped at 16384 lanes by
# the XPU backend, which limits a program to 8192 complex elements.
_CONJ_TILE_LARGE = 8192
_CONJ_TILE_LARGE_FROM = 1 << 18


def conj_blocksize_heur(args):
    n_elements = args['n_elements']
    if n_elements <= 512:
        return 256
    if n_elements <= 4096:
        return 512
    if n_elements < _CONJ_TILE_LARGE_FROM:
        return 1024
    return 4096


def conj_blocks_per_program_heur(args):
    n_elements = args['n_elements']
    if n_elements <= 4096:
        return 1
    if n_elements < _CONJ_TILE_LARGE_FROM:
        return 4
    return _CONJ_TILE_LARGE // 4096


HEURISTICS_CONFIGS = {
    'elementwise_unary': {
        'BLOCK_SIZE': unary_blocksize_heur,
        'BLOCKS_PER_PROGRAM': unary_blocks_per_program_heur,
    },
    'elementwise_conj': {
        'BLOCK_SIZE': conj_blocksize_heur,
        'BLOCKS_PER_PROGRAM': conj_blocks_per_program_heur,
    },
    'elementwise_binary': {
        'BLOCK_SIZE': simple_elementwise_blocksize_heur,
        'BLOCKS_PER_PROGRAM': simple_elementwise_blocks_per_program_heur,
    },
    'elementwise_trinary': {
        'BLOCK_SIZE': simple_elementwise_blocksize_heur,
        'BLOCKS_PER_PROGRAM': simple_elementwise_blocks_per_program_heur,
    },
}
