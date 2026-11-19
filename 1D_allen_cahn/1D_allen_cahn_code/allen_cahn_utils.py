"""
Shared helpers for the 1D Allen-Cahn benchmark.

    u_t - D u_xx + u^3 - u = f(x, t),  D = 1e-4,  x in [-1, 1] (periodic),  t in [0, 1]

Exact solution (the source term f is obtained by substituting it into the equation,
as for the other benchmarks):

    u(x, t) = tanh( kappa(t) cos(pi x) ),   kappa(t) = KAPPA0 + (KAPPA1 - KAPPA0) t

At t = 0 the profile is smooth; as t grows, two interfaces at x = +-1/2 sharpen
(width ~ 1/(pi kappa)) between the plateaus u = +-1, the typical Allen-Cahn
metastable pattern. The solution is 2-periodic in x with all derivatives.

The test set is a uniform grid of N_X x N_T points (x first, meshgrid indexing='ij').
"""
import numpy as np
import torch

D = 1e-4            # diffusion coefficient
REACTION = 1.0      # reaction coefficient: u^3 - u
KAPPA0 = 1.0        # interface sharpness at t = 0
KAPPA1 = 10.0       # interface sharpness at t = 1
N_X, N_T = 512, 201  # test grid


def kappa(t):
    return KAPPA0 + (KAPPA1 - KAPPA0) * t


def allen_cahn_exact(x, t):
    """Exact solution u(x, t); works for torch tensors and numpy arrays."""
    if isinstance(x, torch.Tensor):
        return torch.tanh(kappa(t) * torch.cos(np.pi * x))
    return np.tanh(kappa(t) * np.cos(np.pi * x))


def allen_cahn_source(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Source term f = u_t - D u_xx + u^3 - u of the exact solution (closed form, differentiable)."""
    c = torch.cos(np.pi * x)
    s = torch.sin(np.pi * x)
    k = kappa(t)
    u = torch.tanh(k * c)
    sech2 = 1.0 - u ** 2
    u_t = sech2 * (KAPPA1 - KAPPA0) * c
    u_xx = -np.pi ** 2 * k * sech2 * (2.0 * k * u * s ** 2 + c)
    return u_t - D * u_xx + REACTION * u ** 3 - REACTION * u


def allen_cahn_initial_condition(x: torch.Tensor) -> torch.Tensor:
    """u(x, 0) = tanh(KAPPA0 cos(pi x))"""
    return torch.tanh(KAPPA0 * torch.cos(np.pi * x))


def periodic_embedding(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(x, t) -> (cos(pi x), sin(pi x), t): enforces 2-periodicity in x exactly."""
    return torch.cat([torch.cos(np.pi * x), torch.sin(np.pi * x), t], dim=1)


# ---- test grid ----
def reference_axes() -> tuple:
    """1D axes of the test grid: x (N_X,), t (N_T,)."""
    return np.linspace(-1.0, 1.0, N_X), np.linspace(0.0, 1.0, N_T)


def load_reference_solution() -> tuple:
    """x (N_X,), t (N_T,), u (N_T, N_X) of the exact solution on the test grid."""
    x, t = reference_axes()
    u = allen_cahn_exact(x[None, :], t[:, None])
    return x, t, u


def reference_grid_shape() -> tuple:
    """(n_x, n_t) of the test grid, i.e. the shape used to reshape flattened test data."""
    return N_X, N_T


def reference_test_data() -> tuple:
    """Test points, flattened with meshgrid(x, t, indexing='ij'): (N_X*N_T, 1) float32 tensors."""
    x, t = reference_axes()
    x_grid, t_grid = torch.meshgrid(torch.tensor(x, dtype=torch.float32),
                                    torch.tensor(t, dtype=torch.float32), indexing='ij')
    return x_grid.reshape(-1, 1), t_grid.reshape(-1, 1)


def reference_exact_solution() -> torch.Tensor:
    """Exact u on the test grid, aligned with reference_test_data(): (N_X*N_T, 1) float32 tensor."""
    x_test, t_test = reference_test_data()
    return allen_cahn_exact(x_test, t_test)
