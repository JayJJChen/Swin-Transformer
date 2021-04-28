import torch
import torch.nn as nn


def window_partition_3d(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size, window_size,
        H // window_size, window_size,
        W // window_size, window_size,
        C
    )
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, window_size, window_size, window_size, C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        D (int): Depth of image
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D * H * W /
            window_size / window_size / window_size))
    x = windows.view(
        B,
        D // window_size,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        window_size,
        -1
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


if __name__ == "__main__":
    b = 4
    d = 12
    h = 12
    w = 12
    c = 8
    window_size = 3

    x = torch.randn(b, d, h, w, c)

    print("x size:", x.shape)

    x_part = window_partition_3d(x, window_size)
    print("x_part size:", x_part.shape)

    x_res = window_reverse_3d(x_part, window_size, d, h, w)
    print("x_res size:", x_res.shape)

    print("x == x_res?", torch.equal(x, x_res))
