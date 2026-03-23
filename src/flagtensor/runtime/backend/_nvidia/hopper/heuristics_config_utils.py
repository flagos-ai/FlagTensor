def simple_elementwise_blocksize_heur(args):
    n_elements = args['n_elements']
    if n_elements <= 2048:
        return 512
    return 1024


def simple_elementwise_blocks_per_program_heur(args):
    n_elements = args['n_elements']
    if n_elements <= (1 << 15):
        return 1
    if n_elements <= (1 << 21):
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
}
