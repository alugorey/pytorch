# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import os
import torch
import sys


from torch.profiler import profile, ProfilerActivity

parser = argparse.ArgumentParser()
parser.add_argument("--enable-tuning", action="store_true")
parser.add_argument("--enable-profiler", action="store_true")
parser.add_argument("--enable-cudagraph", action="store_true")
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
)
parser.add_argument("--tt", action="store_true")
parser.add_argument("--tt_eye", action="store_true")
parser.add_argument("--tt_int", action="store_true")
parser.add_argument("--nn", action="store_true")
parser.add_argument("--nn_int", action="store_true")
parser.add_argument("--nn_eye", action="store_true")
parser.add_argument("--tn_eye", action="store_true")
parser.add_argument("--tn", action="store_true")
parser.add_argument("--nt", action="store_true")
parser.add_argument("--tn_km", action="store_true")
parser.add_argument("--nt_eye", action="store_true")
parser.add_argument("--use-ck-gemm", action="store_true")
#parser.add_argument("--small", action="store_true")
#parser.add_argument("--medium", action="store_true")
#parser.add_argument("--large", action="store_true")

#TODO REMOVE THESE
parser.add_argument("--med-gemm-nn", action="store_true")
parser.add_argument("--med-gemm-nt", action="store_true")
parser.add_argument("--med-gemm-tn", action="store_true")
parser.add_argument("--med-gemm-tt", action="store_true")

parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
parser.add_argument("--size", default="small", choices=["small", "medium", "large"])
args = parser.parse_args()

os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
if args.enable_tuning:
    print("Enabled tuning")
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "hipblas_tuning_pt_llama.csv"
    os.environ["PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS"] = "30"
    os.environ["PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS"] = "30"

dtype_mapping = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

dtype_atol_mapping = {
    torch.float32: 0.0008,
    torch.float16: 0.05,
    torch.bfloat16: 0.5,
}

dtype_rtol_mapping = {
    torch.float32: 0.0008,
    torch.float16: 0.005,
    torch.bfloat16: 0.01,
}



shapes = [
    (8192, 1024), # now works
#    (7168, 8192), # now works
#    (8192, 3584), #worked in the past
#    (1280, 8192),
#     (8, 8), # my case that matches example
#     (4, 4) # now works
]

dtype_size_mapping = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}
# TT Shapes
shapes_tt_sm = [
    (8, 64, 8)
]

shapes_tt_md = [
    (1024, 8192, 1024)
]

shapes_tt_lg = [
    (8192, 4096, 128)
]

# TN Shapes
shapes_tn_sm = [
    (8, 8, 64)
]

shapes_tn_md = [
    (1024, 1024, 512)
]

shapes_tn_lg = [
    (8192, 8192, 4096)
]


# NT Shapes
shapes_nt_sm = [
    (8, 64, 1024)
]

shapes_nt_md = [
    (1024, 2048, 1024)
]

shapes_nt_lg = [
    (4096, 8192, 1024)
]

# NN Shapes
shapes_nn_sm = [
    (8, 64, 1024)
]

shapes_nn_md = [
    (1024, 2048, 1024)
]

shapes_nn_lg = [
    (4096, 8192, 1024)
]

#dtype = torch.bfloat16
mytype = dtype_mapping[args.dtype]

print("LUGO dtype: ", mytype)
print("LUGO dtype type", type(mytype))
dtype = dtype_mapping[args.dtype]

do_profile = args.enable_profiler
enable_cudagraph = args.enable_cudagraph
dtype_size = dtype_size_mapping[dtype]

if args.use_ck_gemm:
    torch.backends.cuda.preferred_blas_library('ck')
else:
    print()
    print()
    print("STOP!!! YOU ARE NOW RUNNING THE NON-CK PATH BE CAREFUL MOVING FORWARD")
    print()
    print()
#torch.set_printoptions(profile="full")

shapes_mapping = {
    "tt": {
        "small" : shapes_tt_sm,
        "medium": shapes_tt_md,
        "large" : shapes_tt_lg,
    },
    "tn": {
        "small" : shapes_tn_sm,
        "medium": shapes_tn_md,
        "large" : shapes_tn_lg,
    },
    "nt": {
        "small" : shapes_nt_sm,
        "medium": shapes_nt_md,
        "large" : shapes_nt_lg,
    },
    "nn": {
        "small" : shapes_nn_sm,
        "medium": shapes_nn_md,
        "large" : shapes_nn_lg,
    },
}
results = {}

if args.tt:
    shapes = shapes_mapping["tt"][args.size]
    #shapes = shapes_tt

if args.tn:
    shapes = shapes_mapping["tn"][args.size]
    #shapes = shapes_tn

if args.nt:
    shapes = shapes_mapping["nt"][args.size]
    #shapes = shapes_nt

if args.nn:
    shapes= shapes_mapping["nn"][args.size]
    #shapes = shapes_nn

for (m, n, k) in shapes:
#for (n, k) in shapes:
    print("entered the loop!")
#    m = args.batch_size
    print(f"- Run Linear (matmul) {m} x {n} x {k}, dtype = {dtype}")
    '''
    inp = torch.randn((m, k), dtype=torch.float32, device="cuda")
    weights = torch.randn((n, k), dtype=torch.float32, device="cuda")
    inp_cpu = inp.cpu().to(torch.float)
    weights_cpu = weights.cpu().to(torch.float)
    '''

    # [   M   |   N   |   K   ]
    # [   8   |  8192 |  1024 ]

    if args.nn: # TT | CCR
        inp = torch.randn((m, k), dtype=dtype, device="cuda")
        weights = torch.randn((k, n), dtype=dtype, device="cuda")
        #weights = torch.permute(weights, (1,0))
    elif args.nn_eye: # TT | CCR
        inp = torch.eye(m,k, dtype=dtype, device="cuda")
        weights = torch.randn((k, n), dtype=dtype, device="cuda")
    elif args.nn_int: # TT | CCR
        inp = torch.eye(m,k, dtype=dtype, device="cuda")
        weights = torch.randint(2,5, (k,n), dtype=dtype, device="cuda")


    elif args.tt: # NN CK DOES NOT SUPPORT
        print("TT case")

        inp = torch.randn((m, k), dtype=dtype, device="cuda")
        inp = torch.permute(inp, (1,0))
        weights = torch.randn((n, k), dtype=dtype, device="cuda")
        weights = torch.permute(weights, (1,0))
        print("EXITING TT CASE")

    elif args.tt_eye: # NN | RRR


        #inp = torch.randn((m, k), dtype=dtype, device="cuda")
        inp = torch.eye(m, k, dtype=dtype, device="cuda")
        #print("INPUT SHAPE: ", inp.size())
        inp = torch.permute(inp, (1,0))
        #print("HERE ANDY")
        #print(inp)
        


        weights = torch.randn((n, k), dtype=dtype, device="cuda")
        #weights = torch.eye(n, k, dtype=dtype, device="cuda")
        weights = torch.permute(weights, (1,0))

    elif args.tt_int: # NN | RRR
        #inp = torch.randint(2, 5,(m, k), dtype=dtype, device="cuda")
        inp = torch.eye(m, k, dtype=dtype, device="cuda")
        inp = torch.permute(inp, (1,0))
        


        weights = torch.randint(2, 5, (n, k), dtype=dtype, device="cuda")
        weights = torch.permute(weights, (1,0))


    elif args.tn_eye: #NT | CRR
        inp = torch.eye(k, m, dtype=dtype, device="cuda") # row
        inp = torch.permute(inp, (1,0)) # col
        weights = torch.randn((k, n), dtype=dtype, device="cuda") # row
        #weights = torch.permute(weights, (1,0))


    elif args.tn: # NT | CRR CK DOES NOT SUPPORT
        inp = torch.randn((m, k), dtype=dtype, device="cuda") # row
        inp = torch.permute(inp, (1,0)) # col
        weights = torch.randn((n, k), dtype=dtype, device="cuda") # row


    elif args.tn_km: # NT | CRR
        inp = torch.randn((k, m), dtype=dtype, device="cuda") # row
        inp = torch.permute(inp, (1,0)) # col
        weights = torch.randn((k, n), dtype=dtype, device="cuda") # row


    elif args.nt_eye: # TN | RCR
        inp = torch.eye(m, k, dtype=dtype, device="cuda") # row
        weights = torch.randn((n, k), dtype=dtype, device="cuda") # row
        weights = torch.permute(weights, (1,0)) # col


    elif args.nt: # TN | RCR
        inp = torch.randn((m, k), dtype=dtype, device="cuda")
        weights = torch.randn((n, k), dtype=dtype, device="cuda")
        weights = torch.permute(weights, (1,0))

    elif args.med_gemm_nn:
        print("MEDIUM SIZED GEMM")
        #Avg time: 2552.4252319335938 us, Achieved 3.79 TFLOPS, 10.27 GB/s 
        m = 2048
        n = 1152
        k = 2048

        print(f"- Run Linear (matmul) {m} x {n} x {k}, dtype = {dtype}")
        inp = torch.randn((m, k), dtype=dtype, device="cuda")
        weights = torch.randn((k, n), dtype=dtype, device="cuda")

    elif args.med_gemm_nt:
        #Avg time: 2953.245849609375 us, Achieved 3.27 TFLOPS, 8.88 GB/s
        m = 2048
        n = 1152
        k = 2048

        print(f"- Run Linear (matmul) {m} x {n} x {k}, dtype = {dtype}")
        inp = torch.randn((m, k), dtype=dtype, device="cuda")
        weights = torch.randn((n, k), dtype=dtype, device="cuda")
        weights = torch.permute(weights, (1,0))

    elif args.med_gemm_tn:
        #Avg time: 2006.623077392578 us, Achieved 4.82 TFLOPS, 11.23 GB/s
        m = 2048
        n = 2048
        k = 1152

        print(f"- Run Linear (matmul) {m} x {n} x {k}, dtype = {dtype}")
        inp = torch.randn((m, k), dtype=dtype, device="cuda") # row
        inp = torch.permute(inp, (1,0)) # col
        weights = torch.randn((n, k), dtype=dtype, device="cuda") # row
    elif args.med_gemm_tt:
        #Avg time: 2849.5462036132812 us, Achieved 3.39 TFLOPS, 9.20 GB/s
        m = 2048
        n = 1152
        k = 2048

        print(f"- Run Linear (matmul) {m} x {n} x {k}, dtype = {dtype}")
        inp = torch.randn((m, k), dtype=dtype, device="cuda")
        inp = torch.permute(inp, (1,0))
        weights = torch.randn((n, k), dtype=dtype, device="cuda")
        weights = torch.permute(weights, (1,0))
    else:
        print("No case chosen, exiting...")
        sys.exit()

    #inp = torch.full((m,k), 2.0, dtype=dtype, device="cuda")
    #weights = torch.full((n,k), 4.0, dtype=dtype, device="cuda")
    inp_cpu = inp.cpu().to(torch.float)
    weights_cpu = weights.cpu().to(torch.float)
    print("MADE IT HERE")
    if enable_cudagraph:
        s = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()

        # this may be needed for tuning
        ref = inp @ weights.T
        ref_cpu = inp_cpu @ weights_cpu.T
        s.wait_stream(torch.cuda.current_stream())

        with torch.cuda.graph(g, stream=s):
            for _ in range(100):
                print("IF YOU SEE THIS, YOU DONE GOOFED")
                ref = inp @ weights.T
                ref_cpu = inp_cpu @ weights_cpu.T

    ## Run warmup
    if not enable_cudagraph:
        for _ in range(20):
            #ref = F.linear(inp, weights)
            ref = inp @ weights
            ref_cpu = inp_cpu @ weights_cpu

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()

    if do_profile:
        torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        torch_profiler.start()

    start_event.record()
    if enable_cudagraph:
        g.replay()
    else:
        for _ in range(100):
            '''
            ref = inp @ weights.T
            ref_cpu = inp_cpu @ weights_cpu.T
            '''
            ref = inp @ weights

    stop_event.record()

    ref_cpu = inp_cpu @ weights_cpu
    torch.cuda.synchronize()
    if do_profile:
        torch_profiler.stop()
        torch_profiler.export_chrome_trace(f"{m}_{n}_{k}.json")
    elapsed = start_event.elapsed_time(stop_event)
    ms = elapsed / 100
    us = ms * 1000


#    print("TESTING FOR ACCURACY")
#    print("EXPECTED: ")
#    print(ref_cpu)
#    print("ACTUAL: ")
#    print(ref.cpu())

    '''
    print(torch.testing.assert_close(ref_cpu,
                                     ref.cpu(),
                                     atol=dtype_atol_mapping[dtype],
                                     rtol=dtype_rtol_mapping[dtype]))
    
    
    '''
    print(torch.testing.assert_close(ref_cpu.to(torch.float32),
                                     ref.cpu().to(torch.float32),
                                     atol=dtype_atol_mapping[dtype],
                                     rtol=dtype_rtol_mapping[dtype]))
    print("DONE")

    def compute_FC_flops(m, n, k):
        flops = m * n * k * 2
        return flops

    def compute_total_bytes(m, n, k):
        return (
            dtype_size * m * n
            + dtype_size * n * k
            + dtype_size_mapping[torch.float32] * m * k
        )

    flops = compute_FC_flops(m, n, k) / (ms / 1e3)
    bw = compute_total_bytes(m, n, k) / (ms / 1e3)
    print(
        "Avg time: {} us, Achieved {:.2f} TFLOPS, {:.2f} GB/s\n".format(
            us, flops / 1e12, bw / 1e9
        )
    )
    results[f"{m}x{n}x{k}-{dtype}"] = [us, flops / 1e12, bw / 1e9]

for config, result in results.items():
    out_str = f"{config}"
    for i in result:
        out_str += f",{i}"

    print(out_str)

