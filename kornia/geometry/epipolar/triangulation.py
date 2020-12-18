"""Module with the functionalites for triangulation."""

import torch
import kornia

try:
    import torch_batch_svd
except ImportError:
    torch_batch_svd = None

def svd_wrapper(tensor):
    if 'cuda' not in str(tensor.device):
        return torch.svd(tensor)

    if torch_batch_svd is None:
        U, s, V = torch.svd(tensor.cpu())
        return U.to(tensor.device), s.to(tensor.device), V.to(tensor.device)

    head = lambda t: t.shape[:-2]
    tail = lambda t: t.shape[-2:]

    U, s, V =  torch_batch_svd.svd(tensor.reshape(-1, *tail(tensor)))
    U = U.reshape(*head(tensor), *tail(U))
    s = s.reshape(*head(tensor), -1)
    V = V.reshape(*head(tensor), *tail(V))

    return U, s, V

# https://github.com/opencv/opencv_contrib/blob/master/modules/sfm/src/triangulation.cpp#L68

def triangulate_points(P1: torch.Tensor, P2: torch.Tensor, points1: torch.Tensor,
                       points2: torch.Tensor) -> torch.Tensor:
    r"""Reconstructs a bunch of points by triangulation.

    Triangulates the 3d position of 2d correspondences between several images.
    Reference: Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312

    The input points are assumend to be in homogeneous coordinate system and being inliers
    correspondences. The method does not perform any robust estimation.

    Args:
        P1 (torch.Tensor): The projection matrix for the first camera with shape :math:`(*, 3, 4)`.
        P2 (torch.Tensor): The projection matrix for the second camera with shape :math:`(*, 3, 4)`.
        points1 (torch.Tensor): The set of points seen from the first camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.
        points2 (torch.Tensor): The set of points seen from the second camera frame in the camera plane
          coordinates with shape :math:`(*, N, 2)`.

    Returns:
        torch.Tensor: The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.

    """
    assert len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4), P1.shape
    assert len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4), P2.shape
    assert len(P1.shape[:-2]) == len(P2.shape[:-2]), (P1.shape, P2.shape)
    assert len(points1.shape) >= 2 and points1.shape[-1] == 2, points1.shape
    assert len(points2.shape) >= 2 and points2.shape[-1] == 2, points2.shape
    assert len(points1.shape[:-2]) == len(points2.shape[:-2]), (points1.shape, points2.shape)
    assert len(P1.shape[:-2]) == len(points1.shape[:-2]), (P1.shape, points1.shape)

    # allocate and construct the equations matrix with shape (*, 4, 4)
    points_shape = max(points1.shape, points2.shape)  # this allows broadcasting
    X = torch.zeros(points_shape[:-1] + (4, 4)).type_as(points1)

    for i in range(4):
        X[..., 0, i] = points1[..., 0] * P1[..., 2:3, i] - P1[..., 0:1, i]
        X[..., 1, i] = points1[..., 1] * P1[..., 2:3, i] - P1[..., 1:2, i]
        X[..., 2, i] = points2[..., 0] * P2[..., 2:3, i] - P2[..., 0:1, i]
        X[..., 3, i] = points2[..., 1] * P2[..., 2:3, i] - P2[..., 1:2, i]

    # 1. Solve the system Ax=0 with smallest eigenvalue
    # 2. Return homogeneous coordinates

    U, S, V = svd_wrapper(X)

    points3d_h = V[..., -1]
    points3d: torch.Tensor = kornia.convert_points_from_homogeneous(points3d_h)
    return points3d
