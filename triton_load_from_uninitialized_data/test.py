import torch
import triton
import triton.language as tl


@triton.jit
def kernel(X, n_cols, BLOCK_SIZE:tl.constexpr):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    temp = tl.load(X + cols, mask=mask, other=0.0)
    tl.store(X + cols, temp, mask=mask)

def launch_kernel(input, n_rows, n_cols):
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    kernel[(n_rows,)](input, n_cols, BLOCK_SIZE)

n_times = 100
for i in range(n_times):
    n_rows = 10000
    n_cols = 2000
    input = torch.empty((n_rows, n_cols), dtype = torch.float32, device='cuda')
    print("Before:", torch.max(input))
    launch_kernel(input, n_rows, n_cols)
    print("After:", torch.max(input))


#torch.allclose(input, torch.zeros_like(input), atol=0)