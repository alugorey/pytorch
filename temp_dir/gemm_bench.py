import torch
import argparse



parser = argparse.ArgumentParser()


parser.add_argument("--use-ck", action="store_true")


args = parser.parse_args()

# ----------------------------
# Config
# ----------------------------
SHAPES = [
#    (64, 512, 512),         # small (M <= 128)
#    (512, 512, 512),        # medium
#    (2048, 1024, 1536),     # medium rectangular
#    (4096, 8192, 2048),     # large (N >= 8192 and M >= 4096)
    (4096, 8192, 4096),     # large (M >= 8192 and N >= 4096)
]
LAYOUTS = ["NN", "NT", "TN", "TT"]
#LAYOUTS = ["TT"]
SEED = 17

if args.use_ck:
    torch.backends.cuda.preferred_blas_library('ck')

# default iteration configs (you can tune these)
SMALL_ITERS = dict(warmup=10, iters=100)
MEDIUM_ITERS = dict(warmup=8,  iters=30)
LARGE_ITERS = dict(warmup=4,  iters=10)


def classify_size(M, N):
    """Return 'small', 'medium', or 'large' based on thresholds:
       small:  M <= 128
       large:  (M >= 8192 and N >= 4096) or (N >= 8192 and M >= 4096)
       medium: otherwise
    """
    is_small = (M <= 128)
    is_large = ( (M >= 8192 and N >= 4096) or (N >= 8192 and M >= 4096) )
    if is_small:
        return "small"
    if is_large:
        return "large"
    return "medium"


def iters_for_size(size_class):
    if size_class == "small":
        return SMALL_ITERS["warmup"], SMALL_ITERS["iters"]
    if size_class == "large":
        return LARGE_ITERS["warmup"], LARGE_ITERS["iters"]
    return MEDIUM_ITERS["warmup"], MEDIUM_ITERS["iters"]


def make_inputs(M, N, K, layout, device):
    """
    Layouts:
      NN: A[M,K]   @ B[K,N]
      NT: A[M,K]   @ (B[N,K]).t()
      TN: (A[K,M]).t() @ B[K,N]
      TT: (A[K,M]).t() @ (B[N,K]).t()
    """
    dtype = torch.bfloat16

    if layout == "NN":
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(K, N, device=device, dtype=dtype)
        Aop, Bop = A, B
    elif layout == "NT":
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(N, K, device=device, dtype=dtype)
        Aop, Bop = A, B.t()
    elif layout == "TN":
        A = torch.randn(K, M, device=device, dtype=dtype)
        B = torch.randn(K, N, device=device, dtype=dtype)
        Aop, Bop = A.t(), B
    elif layout == "TT":
        A = torch.randn(K, M, device=device, dtype=dtype)
        B = torch.randn(N, K, device=device, dtype=dtype)
        Aop, Bop = A.t(), B.t()
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return Aop, Bop


@torch.no_grad()
def time_gemm(Aop, Bop, n_warmup, n_iters):
    # warmup
    for _ in range(n_warmup):
        _ = Aop @ Bop

    # timed runs — CUDA events
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_ms = 0.0
    for _ in range(n_iters):
        start.record()
        _ = Aop @ Bop
        end.record()
        torch.cuda.synchronize()
        total_ms += start.elapsed_time(end)  # milliseconds

    avg_us = (total_ms / n_iters) * 1000.0
    return avg_us


def flops_gemm(M, N, K):
    # ~2*M*N*K FLOPs for GEMM
    return 2.0 * M * N * K


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Please run on a CUDA-capable machine.")
        return

    torch.manual_seed(SEED)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # okay with bf16

    cap = torch.cuda.get_device_capability(device)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device : {torch.cuda.get_device_name(device)} (SM {cap[0]}.{cap[1]})")
    print("Dtype  : bfloat16")
    print("Size thresholds:")
    print("  - small : M <= 128")
    print("  - large : (M >= 8192 and N >= 4096) or (N >= 8192 and M >= 4096)")
    print("  - medium: otherwise")
    print("-" * 114)
    print(f"{'M':>6} {'N':>6} {'K':>6}  {'Layout':>6}  {'Size':>7}  "
          f"{'Warm':>5} {'Iters':>5}  {'Avg (µs)':>12}  {'TFLOP/s (est)':>14}")
    print("-" * 114)

    for (M, N, K) in SHAPES:
        size_cls = classify_size(M, N)
        n_warm, n_iter = iters_for_size(size_cls)

        for layout in LAYOUTS:
            Aop, Bop = make_inputs(M, N, K, layout, device)
            avg_us = time_gemm(Aop, Bop, n_warm, n_iter)

            flops = flops_gemm(M, N, K)
            avg_s = avg_us * 1e-6
            tflops = (flops / avg_s) / 1e12

            print(f"{M:6d} {N:6d} {K:6d}  {layout:>6}  {size_cls:>7}  "
                  f"{n_warm:5d} {n_iter:5d}  {avg_us:12.2f}  {tflops:14.2f}")

    print("-" * 114)
    print("Note: Times are average per GEMM using CUDA events; higher TFLOP/s indicates better throughput.")


if __name__ == "__main__":
    main()

