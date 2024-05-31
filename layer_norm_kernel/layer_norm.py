
import torch

import triton
import triton.language as tl


################################################################################
# LayerNorm forward formula
# for each row x in X:
# y = (x - mean) / sqrt(var + eps) . w + b
# mean = sum(X) / n_cols
# var = sum((X - mean)^2) / n_cols
# w has shape (,n_cols)
# b has shape (,n_cols)
################################################################################
configs_layer_norm =[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ]
@triton.autotune(
    configs=configs_layer_norm,
    key=["n_cols"],
)
@triton.jit
def _layer_norm_fwd_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x,  # how much to increase the pointer when moving by 1 row
    stride_y,
    n_cols,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x
    Y += row * stride_y
    # Compute the mean and variance
    # n_cols <= BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    tl.store(Mean + row, mean, mask=mask)
    var = tl.sum((x - mean) ** 2, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd, mask=mask)
    # Compute the output
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * w + b
    tl.store(Y + cols, y, mask=mask)
    
def layer_norm_fwd(
    X,  # input tensor
    W,  # weights
    B,  # biases
    eps,  # epsilon to avoid division by zero
):

    n_rows, n_cols = X.shape

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // X.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))

    # Allocate Mean, Rstd and Y
    Mean = torch.empty((n_rows, ), dtype=torch.float32, device=X.device)
    Rstd = torch.empty((n_rows, ), dtype=torch.float32, device=X.device)
    Y = torch.empty_like(X, dtype=X.dtype, device=X.device)
    
    # assert contiguous memory
    assert X.stride(-1) == 1
    assert Y.stride(-1) == 1
    assert W.stride(-1) == 1
    assert B.stride(-1) == 1
    

    # Call the kernel
    _layer_norm_fwd_kernel[(n_rows,)](
        X, Y, W, B, Mean, Rstd, X.stride(0), Y.stride(0), n_cols, eps, BLOCK_SIZE
    )
    return Y, Mean, Rstd
