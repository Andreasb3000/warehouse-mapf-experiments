#!/usr/bin/env python3
"""Small paper-inspired ConvLSTM baselines for the fixed-floor forecast benchmark.

This is deliberately an adaptation of Yu and Wolf's spatial recurrent model.
It predicts occupancy and low-speed occupancy, not their excess travel-time label.
All model selection uses validation workloads; testing is prediction only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = {
    "name": "two-layer ConvLSTM, direct three-window head",
    "history_offsets_s": [-10, -8, -6, -4, -2, 0],
    "future_windows_s": [[0, 5], [5, 10], [10, 20]],
    "input_channels": ["occupancy", "low_speed_occupancy", "road", "service",
                       "outer_overlap", "is_plan", "normalized_time"],
    "hidden_channels": [4, 4], "kernels": [5, 3],
    "output_channels": ["mean_occupancy", "mean_low_speed_occupancy"],
    "sequence_steps": 9,
    "differences_from_paper": [
        "occupancy and low-speed occupancy targets replace isolated-reference delay",
        "one shared recurrent model with a direct multi-window head",
        "4-channel hidden states; no batch normalization",
        "deterministic current-plan occupancy instead of Gaussian path uncertainty",
        "residual variant is an additional project-specific hybrid",
    ],
}


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden: int, kernel: int):
        super().__init__()
        self.hidden = hidden
        self.gates = nn.Conv2d(input_channels + hidden, 4 * hidden, kernel,
                               padding=kernel // 2)

    def forward(self, x, state):
        if state is None:
            h = x.new_zeros((x.shape[0], self.hidden, *x.shape[-2:]))
            c = torch.zeros_like(h)
        else:
            h, c = state
        i, f, o, g = self.gates(torch.cat((x, h), dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class SpatialPredictor(nn.Module):
    def __init__(self, *, residual=False, hidden=4):
        super().__init__()
        self.residual, self.hidden = bool(residual), int(hidden)
        self.cells = nn.ModuleList([ConvLSTMCell(7, hidden, 5),
                                    ConvLSTMCell(hidden, hidden, 3)])
        self.head = nn.Conv2d(hidden, 6, 1)
        nn.init.normal_(self.head.weight, mean=0, std=.01)
        with torch.no_grad():
            # All channels are interleaved by window: occupancy, slow fraction.
            self.head.bias[:] = torch.tensor([0 if residual else -3, -3] * 3)

    def forward(self, history, plan, static):
        batch, _, _, height, width = history.shape
        state = [None, None]
        for step in range(9):
            image = history[:, step] if step < 6 else plan[:, step - 6]
            kind = 0 if step < 6 else 1
            stamp = ([-10, -8, -6, -4, -2, 0][step] if step < 6
                     else [2.5, 7.5, 15][step - 6]) / 20
            flags = image.new_empty((batch, 2, height, width))
            flags[:, 0] = kind
            flags[:, 1] = stamp
            x = torch.cat((image, static, flags), dim=1)
            for layer, cell in enumerate(self.cells):
                state[layer] = cell(x, state[layer])
                x = state[layer][0]
        raw = self.head(x).reshape(batch, 3, 2, height, width)
        occupancy_logits = raw[:, :, 0]
        if self.residual:
            # Stable inverse softplus. A zero residual recovers the plan plus
            # a small positive floor, permitting learning into empty cells.
            base = plan[:, :, 0] + .005
            occupancy_logits = occupancy_logits + base + torch.log(-torch.expm1(-base))
        occupancy = F.softplus(occupancy_logits)
        low_speed = occupancy * torch.sigmoid(raw[:, :, 1])
        result = torch.stack((occupancy, low_speed), dim=2)
        return result * static[:, None, :1]


def loss_components(prediction, target, road, roi):
    """Predeclared sparse-target loss, keeping spatial and channel terms explicit.

    Within each channel: 40% all-road MSE, 30% fixed-junction/approach ROI
    MSE, 30% positive-target-cell MSE. Across channels: 30% occupancy and
    70% low-speed occupancy. Positive weighting is training-only; benchmark
    errors must still report ordinary masked RMSE and active-cell subsets.
    """
    squared = (prediction - target).square()
    road = road.to(prediction.device)[None, None, None]
    roi = roi.to(prediction.device)[None, None, None]

    def masked(mask):
        weights = mask.expand_as(squared)
        return (squared * weights).sum(dim=(0, 1, 3, 4)) / weights.sum(
            dim=(0, 1, 3, 4)).clamp_min(1)

    all_road = masked(road)
    region = masked(roi)
    positive = masked(road * (target > 1e-6))
    channel = .4 * all_road + .3 * region + .3 * positive
    weighted = .3 * channel[0] + .7 * channel[1]
    return weighted, {"road_mse": all_road, "roi_mse": region,
                      "positive_mse": positive, "channel_loss": channel}


def sample_records(index):
    records = index.get("samples", index.get("origins")) if isinstance(index, dict) else index
    if not isinstance(records, list):
        raise ValueError("Index must contain a samples list in NPZ row order")
    return records


def split_indices(index, n):
    records = sample_records(index)
    if len(records) != n:
        raise ValueError("Index sample count differs from NPZ row count")
    splits = {name: [] for name in ("train", "validation", "test")}
    workload_splits = {}
    for i, record in enumerate(records):
        split = record["split"]
        split = "validation" if split == "val" else split
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}")
        splits[split].append(i)
        workload = record.get("workload_id", record.get("workload", record.get("workload_seed")))
        if workload is not None:
            previous = workload_splits.setdefault(str(workload), split)
            if previous != split:
                raise ValueError("A workload occurs in multiple data splits")
    if any(not indices for indices in splits.values()):
        raise ValueError("Training, validation, and test splits must be nonempty")
    return {key: np.asarray(value, dtype=np.int64) for key, value in splits.items()}


def validate_arrays(data):
    for key in ("history", "plan", "static", "target", "road", "roi"):
        if key not in data:
            raise ValueError(f"Missing dataset key {key}")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite values in {key}")
    n, time_steps, channels, height, width = data["history"].shape
    shapes = {"history": (n, 6, 2, height, width), "plan": (n, 3, 2, height, width),
              "static": (n, 3, height, width), "target": (n, 3, 2, height, width),
              "road": (height, width), "roi": (height, width)}
    for key, shape in shapes.items():
        if data[key].shape != shape:
            raise ValueError(f"{key} shape {data[key].shape} differs from {shape}")
    if not (data["road"].sum() > 0 and data["roi"].sum() > 0):
        raise ValueError("Road and ROI masks must be nonempty")
    if np.any(data["roi"] > data["road"]):
        raise ValueError("ROI must be a subset of traversable cells")
    if np.any(data["target"] < -1e-6):
        raise ValueError("Targets must be nonnegative")
    if np.any(data["target"][:, :, 1] > data["target"][:, :, 0] + 1e-5):
        raise ValueError("Low-speed occupancy cannot exceed occupancy")
    if not np.allclose(data["static"][:, 0], data["road"]):
        raise ValueError("Every sample static channel 0 must match the road mask")
    return n


def synchronize(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _batches(indices, size):
    for offset in range(0, len(indices), size):
        yield indices[offset:offset + size]


def predict(model, arrays, indices, *, device="cpu", batch_size=8):
    model.eval()
    output = []
    with torch.inference_mode():
        for inds in _batches(indices, batch_size):
            args = [arrays[key][inds].to(device) for key in ("history", "plan", "static")]
            output.append(model(*args).cpu().numpy())
    return np.concatenate(output)


def _validation_loss(model, arrays, indices, road, roi, device, batch_size):
    # Evaluate a single aggregate scalar, rather than averaging batch means of
    # differently sized positive subsets. No labels from the test split enter.
    output = predict(model, arrays, indices, device=device, batch_size=batch_size)
    score, components = loss_components(torch.from_numpy(output), arrays["target"][indices],
                                        road.cpu(), roi.cpu())
    return float(score), {k: v.tolist() for k, v in components.items()}


def train_one(data, index, output, *, residual=False, seed=2026, threads=2,
              epochs=30, patience=5, batch_size=8, learning_rate=.001,
              device="auto", hidden=4):
    if min(threads, epochs, patience, batch_size, hidden) <= 0 or learning_rate <= 0:
        raise ValueError("Training sizes, patience, threads, and learning rate must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    n = validate_arrays(data)
    splits = split_indices(index, n)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    arrays = {key: torch.from_numpy(np.asarray(data[key], dtype=np.float32))
              for key in ("history", "plan", "static", "target")}
    road = torch.from_numpy(np.asarray(data["road"], dtype=np.float32))
    roi = torch.from_numpy(np.asarray(data["roi"], dtype=np.float32))
    model = SpatialPredictor(residual=residual, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    trace, best, best_epoch, stale = [], math.inf, -1, 0
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(splits["train"])
        train_sum = 0.
        epoch_started = time.perf_counter()
        for inds in _batches(order, batch_size):
            args = [arrays[key][inds].to(device) for key in ("history", "plan", "static")]
            target = arrays["target"][inds].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(*args)
            loss, _ = loss_components(predicted, target, road, roi)
            if not torch.isfinite(loss):
                raise ValueError("Training loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * len(inds)
        validation, components = _validation_loss(model, arrays, splits["validation"], road,
                                                   roi, device, batch_size)
        synchronize(device)
        row = {"epoch": epoch + 1, "training_batch_mean_loss": train_sum / len(order),
               "validation_loss": validation, "validation_components": components,
               "wall_s": time.perf_counter() - epoch_started}
        trace.append(row)
        print(json.dumps({"method": "residual_convlstm" if residual else "convlstm", **row}),
              flush=True)
        if validation < best - 1e-8:
            best, best_epoch, stale = validation, epoch + 1, 0
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "hidden": hidden, "residual": residual,
                        "architecture": {**ARCHITECTURE, "hidden_channels": [hidden, hidden]}},
                       output / "model.pt")
        else:
            stale += 1
        if stale >= patience:
            break
    synchronize(device)
    training_wall = time.perf_counter() - started
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    # Predictions for all rows simplify common scoring; rows retain their split.
    prediction = predict(model, arrays, np.arange(n), device=device, batch_size=batch_size)
    np.save(output / "predictions.npy", prediction)
    metadata = {
        "method": "residual_convlstm" if residual else "convlstm",
        "architecture": {**ARCHITECTURE, "hidden_channels": [hidden, hidden]},
        "seed": seed, "device": device, "cpu_threads": threads,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "batch_size": batch_size, "learning_rate": learning_rate,
        "maximum_epochs": epochs, "patience": patience, "best_epoch": best_epoch,
        "epochs_completed": len(trace), "best_validation_loss": best,
        "training_wall_s": training_wall, "trace": trace,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_bytes": (output / "model.pt").stat().st_size,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loss": {"channel_weights": {"occupancy": .3, "low_speed": .7},
                 "mask_weights": {"road": .4, "roi": .3, "positive_truth": .3},
                 "description": "weighted MSE; no test observations used for fitting or selection"},
        "prediction_shape": list(prediction.shape),
    }
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return model, arrays, splits, metadata


def time_forward(model, arrays, indices, *, threads=(1, 2, 4), include_mps=True,
                 warmup=20, repetitions=50):
    """Batch-one inference, synchronized, precomputed features, no training cost.

    CPU input tensors are already resident. MPS reports separately with inputs
    already resident on-device; this excludes host/device transfer by design.
    Rotates up to ten held-out origins instead of repeatedly timing one input.
    """
    original_threads = torch.get_num_threads()
    output = []
    states = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    backends = [("cpu", int(n)) for n in threads]
    if include_mps and torch.backends.mps.is_available():
        backends.append(("mps", original_threads))
    chosen = indices[:min(10, len(indices))]
    for device, count in backends:
        torch.set_num_threads(count)
        candidate = SpatialPredictor(residual=model.residual, hidden=model.hidden).to(device)
        candidate.load_state_dict(states)
        candidate.eval()
        inputs = [[arrays[key][int(i):int(i)+1].to(device)
                   for key in ("history", "plan", "static")] for i in chosen]
        measurements = []
        with torch.inference_mode():
            for step in range(warmup + repetitions):
                synchronize(device)
                start = time.perf_counter()
                candidate(*inputs[step % len(inputs)])
                synchronize(device)
                elapsed = time.perf_counter() - start
                if step >= warmup:
                    measurements.append(elapsed)
        output.append({"device": device, "cpu_threads": count, "batch_size": 1,
                       "warmup": warmup, "repetitions": repetitions,
                       "distinct_test_origins": len(inputs), "mean_s": float(np.mean(measurements)),
                       "median_s": float(np.median(measurements)),
                       "p95_s": float(np.percentile(measurements, 95)),
                       "min_s": min(measurements), "max_s": max(measurements),
                       "samples_s": measurements,
                       "includes": "model forward and device synchronization",
                       "excludes": "feature extraction, loading, host-device transfer, communication"})
    torch.set_num_threads(original_threads)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("convlstm", "residual_convlstm"),
                        default=["convlstm", "residual_convlstm"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--timing-only", action="store_true",
                        help="Benchmark saved checkpoints without fitting; run when host is idle")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = dict(np.load(args.data, allow_pickle=False))
    index = json.loads(args.index.read_text())
    for method in args.methods:
        target = args.output / method
        if args.timing_only:
            checkpoint = torch.load(target / "model.pt", map_location="cpu", weights_only=True)
            model = SpatialPredictor(residual=checkpoint["residual"], hidden=checkpoint["hidden"])
            model.load_state_dict(checkpoint["state_dict"])
            arrays = {key: torch.from_numpy(np.asarray(data[key], dtype=np.float32))
                      for key in ("history", "plan", "static")}
            splits = split_indices(index, len(data["history"]))
            timing = time_forward(model, arrays, splits["test"])
            (target / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
            print(json.dumps({"method": method, "timing": timing}), flush=True)
            continue
        model, arrays, splits, metadata = train_one(
            data, index, target, residual=method == "residual_convlstm", seed=args.seed,
            threads=args.threads, epochs=args.epochs, patience=args.patience,
            batch_size=args.batch_size, device=args.device, hidden=args.hidden)
        metadata["dataset_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
        metadata["index_sha256"] = hashlib.sha256(args.index.read_bytes()).hexdigest()
        (target / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if not args.skip_timing:
            timing = time_forward(model, arrays, splits["test"])
            (target / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
            print(json.dumps({"method": method, "timing": timing}), flush=True)


if __name__ == "__main__":
    main()
