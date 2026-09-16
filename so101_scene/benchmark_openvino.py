"""
Intel inference benchmark script -- required deliverable #3.

Runs the OpenVINO-converted drawer-state vision classifier and reports
latency, throughput, device used, and model precision.

Run on whatever hardware is available:
    python benchmark_openvino.py
    python benchmark_openvino.py --device GPU   # if an Intel iGPU is present
    python benchmark_openvino.py --device NPU   # if an Intel NPU is present

Honest disclosure (see README.md "Known limitations"): the local dev
machine for this project is a 13th Gen Intel Core i7-13650HX (Raptor Lake),
which has NO integrated NPU and no Intel iGPU (discrete RTX 4050 only) --
so CPU is the only device this script can actually exercise there. Real
Core Ultra CPU/iGPU/NPU numbers, if obtained via Intel's free AI PC Cloud,
should be recorded separately and both sets of numbers kept in the
submission -- do not overwrite the honest local-CPU numbers with a
NPU-only number that implies validation happened somewhere it didn't.
"""
import argparse
import time
import numpy as np
import openvino as ov

import os
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "vision_model_ir", "vision_model.xml")
N_WARMUP = 10
N_ITERS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU", help="CPU, GPU, NPU, or AUTO")
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    core = ov.Core()
    available = core.available_devices
    print(f"OpenVINO {ov.__version__}")
    print(f"Available devices on this machine: {available}")

    if args.device not in available and args.device != "AUTO":
        print(f"WARNING: requested device '{args.device}' not available here. "
              f"Falling back to CPU. This machine's actual devices: {available}")
        device = "CPU"
    else:
        device = args.device

    model = core.read_model(args.model)
    compiled = core.compile_model(model, device)

    input_layer = compiled.input(0)
    # Input shape is [None, n_features] (dynamic batch dim) -- static shape
    # access on a dynamic PartialShape throws, so pull the static dim
    # directly rather than calling .shape on the whole thing.
    n_features = input_layer.partial_shape[1].get_length()
    dummy_input = np.random.rand(1, n_features).astype(np.float32)

    # Precision: report what the IR was actually saved as
    precisions = {inp.get_any_name(): inp.get_element_type().get_type_name()
                  for inp in compiled.inputs}

    infer_request = compiled.create_infer_request()

    for _ in range(N_WARMUP):
        infer_request.infer({0: dummy_input})

    latencies = []
    t_start = time.perf_counter()
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        infer_request.infer({0: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000)  # ms
    total_time = time.perf_counter() - t_start

    latencies = np.array(latencies)
    throughput = N_ITERS / total_time

    print("\n=== Benchmark Results ===")
    print(f"Device:            {device}")
    print(f"Model:             {args.model}")
    print(f"Input precision:   {precisions}")
    print(f"Iterations:        {N_ITERS} (after {N_WARMUP} warmup runs)")
    print(f"Latency p50:       {np.percentile(latencies, 50):.4f} ms")
    print(f"Latency p90:       {np.percentile(latencies, 90):.4f} ms")
    print(f"Latency mean:      {latencies.mean():.4f} ms")
    print(f"Throughput:        {throughput:.1f} inferences/sec")
    print()
    print("NOTE: this device list and these numbers reflect THIS machine.")
    print("If this ran on a Core Ultra Series 2/3 system (e.g. via Intel's")
    print("AI PC Cloud), that should be stated explicitly here. If it ran")
    print("on a non-Core-Ultra machine, that should be stated too -- see")
    print("README.md 'Known limitations' for this project's actual hardware.")


if __name__ == "__main__":
    main()