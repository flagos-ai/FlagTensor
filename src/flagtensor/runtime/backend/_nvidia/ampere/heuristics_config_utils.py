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
    if n_elements <= 1024:
        return 256
    if n_elements <= 8192:
        return 512
    return 1024


def simple_elementwise_blocks_per_program_heur(args):
    n_elements = args['n_elements']
    if n_elements <= 8192:
        return 1
    if n_elements <= (1 << 20):
        return 2
    return 4


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
}
