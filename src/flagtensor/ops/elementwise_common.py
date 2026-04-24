import math

import torch


def prepare_elementwise_context(tensors, mode_groups, *, output_modes=None, out=None, device=None, dtype=None):
    normalized_mode_groups = tuple(tuple(modes) for modes in mode_groups)
    tensors_with_modes = tuple(zip(tensors, normalized_mode_groups))
    inferred_output_modes = infer_elementwise_output_modes(normalized_mode_groups, output_modes)
    output_shape = infer_elementwise_output_shape(tensors_with_modes, inferred_output_modes)
    if device is None:
        device = tensors[0].device
    if dtype is None:
        dtype = tensors[0].dtype
    output = allocate_output_tensor(output_shape, out=out, device=device, dtype=dtype)
    return {
        "mode_groups": normalized_mode_groups,
        "output_modes": inferred_output_modes,
        "output_shape": output_shape,
        "output": output,
        "tensors_with_modes": tensors_with_modes,
    }


def infer_elementwise_output_modes(mode_groups, output_modes=None):
    if output_modes is not None:
        normalized = tuple(output_modes)
        if len(set(normalized)) != len(normalized):
            raise ValueError("each output mode may appear at most once")
        return normalized
    ordered = []
    for mode_group in mode_groups:
        for mode in mode_group:
            if mode not in ordered:
                ordered.append(mode)
    return tuple(ordered)


def infer_elementwise_output_shape(tensors_with_modes, output_modes):
    extents = {}
    for tensor, modes in tensors_with_modes:
        for extent, mode in zip(tensor.shape, modes):
            previous = extents.get(mode, 1)
            if previous != 1 and extent != 1 and previous != extent:
                raise ValueError(f"incompatible broadcast extent for mode {mode}: {previous} vs {extent}")
            extents[mode] = max(previous, extent)
    return tuple(extents.get(mode, 1) for mode in output_modes)


def allocate_output_tensor(output_shape, *, out, device, dtype):
    output = out if out is not None else torch.empty(output_shape, device=device, dtype=dtype)
    if tuple(output.shape) != tuple(output_shape):
        raise ValueError(f"output tensor shape mismatch: expected {tuple(output_shape)}, got {tuple(output.shape)}")
    return output


def align_tensor_to_output(tensor: torch.Tensor, tensor_modes, output_modes, output_shape):
    tensor_modes = tuple(tensor_modes)
    output_modes = tuple(output_modes)
    present_modes = [mode for mode in output_modes if mode in tensor_modes]
    permute_order = [tensor_modes.index(mode) for mode in present_modes]
    aligned = tensor
    if permute_order != list(range(len(permute_order))):
        aligned = aligned.permute(*permute_order)
    reshape_shape = []
    present_index = 0
    for mode in output_modes:
        if mode in tensor_modes:
            reshape_shape.append(aligned.shape[present_index])
            present_index += 1
        else:
            reshape_shape.append(1)
    aligned = aligned.reshape(reshape_shape)
    aligned = aligned.expand(output_shape)
    return aligned.contiguous()


def align_tensors_to_output(tensors_with_modes, output_modes, output_shape):
    aligned = []
    for tensor, tensor_modes in tensors_with_modes:
        aligned.append(align_tensor_to_output(tensor, tensor_modes, output_modes, output_shape))
    return tuple(aligned)


def dispatch_elementwise(context, *, try_indexed, execute_aligned):
    indexed = try_indexed(context)
    if indexed is not None:
        return indexed
    aligned_tensors = align_tensors_to_output(
        context["tensors_with_modes"],
        context["output_modes"],
        context["output_shape"],
    )
    return execute_aligned(context, aligned_tensors)


def build_axis_map_and_strides(tensor: torch.Tensor, tensor_modes, output_modes):
    axis_map = []
    stride_map = []
    shape_map = []
    tensor_modes = tuple(tensor_modes)
    for mode in output_modes:
        if mode in tensor_modes:
            axis = tensor_modes.index(mode)
            axis_map.append(axis)
            stride_map.append(tensor.stride()[axis])
            shape_map.append(tensor.shape[axis])
        else:
            axis_map.append(-1)
            stride_map.append(0)
            shape_map.append(1)
    return tuple(axis_map), tuple(stride_map), tuple(shape_map)


def pad_to_rank(values, fill, rank=4):
    return tuple(values) + (fill,) * (rank - len(values))


def _tensor_group_info(tensor, tensor_modes, output_modes, axes):
    tensor_modes = tuple(tensor_modes)
    mapped = []
    for output_axis in axes:
        mode = output_modes[output_axis]
        if mode not in tensor_modes:
            return None
        tensor_axis = tensor_modes.index(mode)
        tensor_shape = tensor.shape[tensor_axis]
        if tensor_shape == 1:
            return None
        tensor_stride = tensor.stride()[tensor_axis]
        mapped.append((output_axis, tensor_axis, tensor_shape, tensor_stride))

    tensor_axes = [item[1] for item in mapped]
    if tensor_axes != sorted(tensor_axes):
        return None
    if tensor_axes[-1] - tensor_axes[0] + 1 != len(tensor_axes):
        return None

    for idx in range(len(mapped) - 1):
        left = mapped[idx]
        right = mapped[idx + 1]
        if left[3] != right[3] * right[2]:
            return None

    group_shape = 1
    for _, _, axis_shape, _ in mapped:
        group_shape *= axis_shape

    base_stride = mapped[-1][3]
    return 0, base_stride, group_shape


def split_output_modes_into_groups(output_shape, tensors_with_modes, output_modes, max_groups=4):
    rank = len(output_shape)
    if rank == 0 or rank > 16:
        return None
    groups = []
    current = [rank - 1]

    def can_merge(candidate_axes):
        for tensor, tensor_modes in tensors_with_modes:
            if _tensor_group_info(tensor, tensor_modes, output_modes, candidate_axes) is None:
                return False
        return True

    for axis in range(rank - 2, -1, -1):
        candidate = [axis] + current
        remaining_axes = axis
        remaining_groups_capacity = max_groups - len(groups) - 1
        if can_merge(candidate) and remaining_axes <= remaining_groups_capacity:
            current = candidate
        else:
            groups.append(current)
            current = [axis]
    groups.append(current)
    groups = list(reversed(groups))
    if len(groups) > max_groups:
        return None
    return [(axes, math.prod(output_shape[axis] for axis in axes)) for axes in groups]


def build_grouped_axis_stride_shape(tensor, tensor_modes, output_modes, groups):
    grouped_axes = []
    grouped_strides = []
    grouped_shapes = []
    for axes, _ in groups:
        info = _tensor_group_info(tensor, tensor_modes, output_modes, axes)
        if info is None:
            return None
        grouped_axes.append(info[0])
        grouped_strides.append(info[1])
        grouped_shapes.append(info[2])
    return tuple(grouped_axes), tuple(grouped_strides), tuple(grouped_shapes)


def prepare_indexed_launch_tensors(output_shape, tensors_with_modes, output_modes, rank=4):
    if len(output_shape) <= rank:
        padded_shape = tuple(output_shape) + (1,) * (rank - len(output_shape))
        metadata = []
        for tensor, tensor_modes in tensors_with_modes:
            axes, strides, shapes = build_axis_map_and_strides(tensor, tensor_modes, output_modes)
            metadata.append(
                (
                    pad_to_rank(axes, -1, rank),
                    pad_to_rank(strides, 0, rank),
                    pad_to_rank(shapes, 1, rank),
                )
            )
        return padded_shape, metadata

    groups = split_output_modes_into_groups(output_shape, tensors_with_modes, output_modes, max_groups=rank)
    if groups is None:
        return None

    metadata = []
    for tensor, tensor_modes in tensors_with_modes:
        grouped = build_grouped_axis_stride_shape(tensor, tensor_modes, output_modes, groups)
        if grouped is None:
            return None
        axes, strides, shapes = grouped
        metadata.append(
            (
                pad_to_rank(axes, -1, rank),
                pad_to_rank(strides, 0, rank),
                pad_to_rank(shapes, 1, rank),
            )
        )

    padded_shape = tuple(extent for _, extent in groups) + (1,) * (rank - len(groups))
    return padded_shape, metadata
