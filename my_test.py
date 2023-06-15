
import torch
from torch.testing._internal_common_utils import TestCase


upper = False
left = True
unitriangular = False


A = torch.tensor([[[-2.7555,  0.0000,  0.0000,  0.0000,  0.0000],
    [-8.9614, -7.4028,  0.0000,  0.0000,  0.0000],
    [-6.2924,  3.3676,  5.8906,  0.0000,  0.0000],
    [ 8.3847, -7.9657,  6.7326, -4.5389,  0.0000],
    [ 0.9497, -7.1549, -5.5168, -7.7067,  8.8918]],
    [[ 8.9637,  0.0000,  0.0000,  0.0000,  0.0000],
    [ 1.3692, -0.2126,  0.0000,  0.0000,  0.0000],
    [ 4.9783, -6.8676, -0.0518,  0.0000,  0.0000],
    [-1.7903,  4.7232,  7.8816, -0.1835,  0.0000],
    [-6.5188, -5.7379,  6.2395,  6.7875,  1.5859]]])

B = torch.tensor([[[ 2.9815e+00],
    [ 7.9675e+00],
    [ 1.6018e-03],
    [-6.8995e+00],
    [ 3.0435e+00]],
    [[ 6.5424e+00],
    [ 8.1279e+00],
    [-6.7546e+00],
    [ 4.9149e+00],
    [ 6.5522e+00]]])

out = B.clone()

torch.linalg.solve_triangular(A, B, upper=upper, left=left, unitriangular=unitriangular, out=out)


TestCase.assertEqual(
