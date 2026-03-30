import csv
import os
from collections import defaultdict
from typing import Iterable

import ast
import matplotlib.pyplot as plt


def write_benchmark_csv(results: Iterable, output_path: str):
    results = list(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["shape", "dtype", "mode", "latency", "latency_base", "speedup"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    mode_groups = defaultdict(list)
    for result in results:
        mode_groups[getattr(result, "mode", "operator")].append(result)

    root, ext = os.path.splitext(output_path)
    for mode, items in mode_groups.items():
        mode_output_path = f"{root}_{mode}{ext}"
        with open(mode_output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in items:
                writer.writerow(result.to_dict())


def plot_latency_and_speedup(results: Iterable, output_dir: str, op_name: str):
    os.makedirs(output_dir, exist_ok=True)
    grouped = defaultdict(list)
    for result in results:
        grouped[(result.dtype, getattr(result, "mode", "operator"))].append(result)

    for (dtype, mode), items in grouped.items():
        items = sorted(
            items,
            key=lambda item: ast.literal_eval(item.shape)[0] if isinstance(item.shape, str) else item.shape[0],
        )
        sizes = [ast.literal_eval(item.shape)[0] if isinstance(item.shape, str) else item.shape[0] for item in items]
        dtype_label = dtype.split(".")[-1] if isinstance(dtype, str) else str(dtype)
        triton_latency = [item.latency for item in items]
        cutensor_latency = [item.latency_base for item in items]
        speedup = [item.speedup for item in items]

        plt.figure(figsize=(8, 5))
        plt.plot(sizes, triton_latency, marker="o", label="Triton")
        plt.plot(sizes, cutensor_latency, marker="o", label="cuTensor")
        plt.xscale("log", base=2)
        plt.xlabel("Tensor size")
        plt.ylabel("Latency (ms)")
        plt.title(f"{op_name} Latency Comparison ({dtype}, {mode})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        latency_path = os.path.join(output_dir, f"{dtype_label}_{mode}_latency.png")
        plt.tight_layout()
        plt.savefig(latency_path, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(sizes, speedup, marker="o", label="Triton vs cuTensor")
        plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)
        plt.xscale("log", base=2)
        plt.xlabel("Tensor size")
        plt.ylabel("Speedup (cuTensor / Triton)")
        plt.title(f"{op_name} Speedup Comparison ({dtype}, {mode})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        speedup_path = os.path.join(output_dir, f"{dtype_label}_{mode}_speedup.png")
        plt.tight_layout()
        plt.savefig(speedup_path, dpi=200)
        plt.close()
