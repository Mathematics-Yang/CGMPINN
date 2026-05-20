import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
import matplotlib.pyplot as plt
import time
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

# set random seeds for reproducibility
torch.manual_seed(1224)
np.random.seed(1224)

# advective-diffusion equation parameters
BETA = 1.0
NU = 1e-2
SPACE_DOMAIN = (-1.0, 1.0)
TIME_DOMAIN = (0.0, 1.0)

class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

class AdaptiveLossWeights:
    """
    self-adaptive loss weights base class
    """
    def __init__(self, n_losses=3, device='cpu'):
        super().__init__()
        self.n_losses = n_losses  # number of loss components
        self.device = device      # device for storing weights
        self.weights = torch.ones(n_losses, device=device, dtype=torch.float32)  # initial weights (equal)

class ReLoBRaLoWeights(AdaptiveLossWeights):
    def __init__(self, n_losses=3, alpha=0.999, temperature=1.0,
                 rho=0.99, device='cpu'):
        super().__init__(n_losses, device)
        self.alpha = alpha           # EMA decay factor
        self.temperature = temperature  # softmax temperature
        self.rho = rho               # random backtracking probability

        # sort of historical losses for random backtracking
        self.loss_history = [[] for _ in range(n_losses)]
        self.ema_losses = None
        self.initial_losses = None

    def update(self, losses, **kwargs):
        losses_tensor = torch.tensor(losses, device=self.device, dtype=torch.float32)

        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss)

        # initialization: set initial losses and EMA to the first observed losses
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.clone()
            self.ema_losses = losses_tensor.clone()
            return

        # update EMA losses
        self.ema_losses = self.alpha * self.ema_losses + (1 - self.alpha) * losses_tensor

        # compute reference losses for weight calculation
        if np.random.rand() < self.rho:
            reference = self.ema_losses
        else:
            # choose a random historical loss for each component
            lookback_idx = np.random.randint(0, max(1, len(self.loss_history[0]) - 1))
            reference = torch.tensor(
                [self.loss_history[i][lookback_idx] for i in range(self.n_losses)],
                device=self.device, dtype=torch.float32
            )

        relative_losses = losses_tensor / (reference + 1e-8)
        scaled_losses = relative_losses / self.temperature
        self.weights = self.n_losses * torch.softmax(scaled_losses, dim=0)

# GMM Curriculum Learning Weight Calculator (from easy to hard)
class GMMCurriculumWeight:
    def __init__(self, n_components=4, update_interval=200, epsilon=1e-6,
                 beta=1.0,   # CL intensity factor
                 tau_saturation=0.8,
                 use_variance_factor=False):
        """
        Initialize GMM Curriculum Learning Weight Calculator (from easy to hard)
        :param n_components: Number of Gaussian components K
        :param update_interval: Weight update interval
        :param epsilon: Small value to prevent division by zero
        :param beta: Intensity coefficient for curriculum learning (larger beta = sharper difficulty transition)
        :param tau_saturation: Tau saturation point (default 0.8, tau reaches 1 at 80% of total steps)
        :param use_variance_factor: Whether to use variance factor for weight optimization
        """
        self.n_components = n_components
        self.update_interval = update_interval
        self.epsilon = epsilon
        self.beta = beta
        self.tau_saturation = tau_saturation
        self.use_variance_factor = use_variance_factor
        self.gmm = GaussianMixture(n_components=n_components, random_state=1224)
        self._last_valid_weights = None  # Initialize valid weight cache

    def compute_weights(self, residuals: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Calculate weights based on residuals and training progress (core: tau controls transition from easy to hard)
        :param residuals: PDE residual tensor (n_pde, 1)
        :param tau: Training progress (0=initial, 1=final)
        :return: Weight tensor for each collocation point (n_pde, 1)
        """
        res_np = residuals.detach().cpu().numpy().reshape(-1, 1)

        try:
            # 1. Fit residual distribution with GMM
            self.gmm.fit(res_np)
            gamma = self.gmm.predict_proba(res_np)  # (n_pde, K) posterior probability
            sigma_sq = self.gmm.covariances_.flatten()  # (K,) variance of each component

            # 2. Calculate component difficulty using posterior probability weighting (more accurate difficulty assessment)
            res_squared = res_np.flatten() ** 2
            component_difficulty = np.array([
                np.sum(gamma[:, j] * res_squared) / (np.sum(gamma[:, j]) + self.epsilon)
                for j in range(self.n_components)
            ])

            # 3. Normalize difficulty to [0, 1] (handle edge cases to prevent division by zero)
            diff_min, diff_max = component_difficulty.min(), component_difficulty.max()
            if diff_max - diff_min > self.epsilon:
                normalized_diff = (component_difficulty - diff_min) / (diff_max - diff_min)
            else:
                normalized_diff = np.zeros_like(component_difficulty)

            # 4. Curriculum learning weights (larger beta increases discrimination)
            easy_weight = np.exp(-self.beta * normalized_diff)  # Easy sample weight: low difficulty → high weight
            hard_weight = np.exp(-self.beta * (1 - normalized_diff))  # Hard sample weight: high difficulty → high weight

            # 5. Smooth transition controlled by tau (from easy to hard)
            curriculum_weight = (1 - tau) * easy_weight + tau * hard_weight

            # 6. Optional variance factor (improves stability in early training, weakens with increasing tau)
            if self.use_variance_factor:
                variance_factor = 1 / (sigma_sq + self.epsilon)
                variance_factor = variance_factor / variance_factor.max()  # Normalize variance factor
                effective_variance_factor = (1 - tau) * variance_factor + tau * 1.0  # Smoothly weaken variance influence
                component_weights = curriculum_weight * effective_variance_factor
            else:
                component_weights = curriculum_weight

            # 7. Sample weights = weighted sum of posterior probability × component weights
            point_weights = np.sum(gamma * component_weights[np.newaxis, :], axis=1)
            self._last_valid_weights = point_weights.copy()  # Update valid weight cache

        except Exception as e:
            print(f"GMM training failed: {e}")
            # Exception handling: use last valid weights or all-one weights
            if self._last_valid_weights is not None and len(self._last_valid_weights) == len(res_np):
                point_weights = self._last_valid_weights
            else:
                point_weights = np.ones(len(res_np))

        # 8. Normalize weights (ensure mean=1, preserve overall loss scale)
        point_weights = point_weights / (point_weights.mean() + self.epsilon)

        # Convert to torch tensor and return (match input device)
        return torch.tensor(point_weights.reshape(-1, 1), dtype=torch.float32).to(residuals.device)

# CGMPINN model
class CGMPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation,
                 use_curriculum=True, gmm_kwargs=None,
                 use_relobralo=True, relobralo_kwargs=None):
        super(CGMPINN, self).__init__()
        self.activation = activation
        self.use_curriculum = use_curriculum  # Curriculum learning switch
        self.use_relobralo = use_relobralo    # ReLoBRaLo adaptive weight switch
        self.total_train_steps = 0  # Record total training steps (for tau calculation)
        self.current_tau = 0.0     # Current training progress

        # Build network structure
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

        # Initialize curriculum learning weight calculator
        if self.use_curriculum:
            gmm_kwargs = gmm_kwargs or {}
            self.curriculum_weight = GMMCurriculumWeight(**gmm_kwargs)
            self.sample_weights = None

        # Cache latest loss components
        self.latest_pde_loss = None
        self.latest_initial_loss = None
        self.latest_boundary_loss = None

        # Initialize ReLoBRaLo adaptive loss weight calculator
        if self.use_relobralo:
            relobralo_kwargs = relobralo_kwargs or {}
            model_device = next(self.parameters()).device
            relobralo_kwargs['device'] = model_device
            self.relobralo = ReLoBRaLoWeights(**relobralo_kwargs)

    def update_curriculum_weights(self, x: torch.Tensor, t: torch.Tensor) -> None:
        """Update curriculum learning weights"""
        if not self.use_curriculum:
            return
        # Calculate advection-diffusion equation residuals
        u_t, u_x, u_xx = self.compute_gradients(x, t)
        pde_residual = u_t + BETA * u_x - NU * u_xx
        # Calculate weights based on current training progress tau
        self.sample_weights = self.curriculum_weight.compute_weights(pde_residual, self.current_tau)

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        """Set training progress tau = current_step / total_steps"""
        if not self.use_curriculum:
            return
        if total_steps == 0:
            self.current_tau = 0.0
        else:
            tau_base = total_steps * self.curriculum_weight.tau_saturation
            self.current_tau = min(current_step / tau_base, 1.0)

    # Forward propagation
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)
        return self.layers(input_tensor)

    # Compute partial derivatives
    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.forward(x, t)
        # First-order time derivative u_t
        u_t = torch.autograd.grad(
            outputs=u, inputs=t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        # First-order spatial derivative u_x
        u_x = torch.autograd.grad(
            outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        # Second-order spatial derivative u_xx
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return u_t, u_x, u_xx

    # Advection-diffusion equation PDE loss
    def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Advection-diffusion equation loss: u_t + βu_x - νu_xx = 0"""
        u_t, u_x, u_xx = self.compute_gradients(x, t)
        pde_residual = u_t + BETA * u_x - NU * u_xx

        if self.use_curriculum and self.sample_weights is not None:
            weighted_residual = self.sample_weights.detach() * (pde_residual ** 2)
            pde_loss_val = torch.mean(weighted_residual)
        else:
            pde_loss_val = torch.mean(pde_residual ** 2)

        self.latest_pde_loss = pde_loss_val.item()
        return pde_loss_val

    # Initial condition loss
    def initial_loss(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        """Initial condition: u(x, 0) = -sin(πx)"""
        u_pred = self.forward(x, t0)
        u_exact = -torch.sin(np.pi * x)  # Analytical initial condition
        initial_residual = u_pred - u_exact
        initial_loss_val = torch.mean(initial_residual ** 2)

        self.latest_initial_loss = initial_loss_val.item()
        return initial_loss_val

    # Boundary condition loss (periodic boundary conditions)
    def boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        """
        Periodic boundary conditions:
        1. u(-1, t) = u(1, t)
        2. ∂u/∂x(-1, t) = ∂u/∂x(1, t)
        """
        # Left boundary x=-1
        x_left = -torch.ones_like(t, requires_grad=True)
        u_left = self.forward(x_left, t)
        # Compute first-order derivative at left boundary
        u_x_left = torch.autograd.grad(
            outputs=u_left, inputs=x_left, grad_outputs=torch.ones_like(u_left),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Right boundary x=1
        x_right = torch.ones_like(t, requires_grad=True)
        u_right = self.forward(x_right, t)
        # Compute first-order derivative at right boundary
        u_x_right = torch.autograd.grad(
            outputs=u_right, inputs=x_right, grad_outputs=torch.ones_like(u_right),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Boundary residuals
        bc1_residual = u_left - u_right  # u(-1,t) = u(1,t)
        bc2_residual = u_x_left - u_x_right  # ∂u/∂x(-1,t) = ∂u/∂x(1,t)

        boundary_loss_val = torch.mean(bc1_residual ** 2) + torch.mean(bc2_residual ** 2)
        self.latest_boundary_loss = boundary_loss_val.item()

        return boundary_loss_val

    def compute_weighted_total_loss(self, pde_data, initial_data, boundary_data) -> torch.Tensor:
        """Compute weighted total loss"""
        x_pde, t_pde = pde_data
        x_initial, t_initial = initial_data
        t_boundary = boundary_data

        pde_loss_val = self.pde_loss(x_pde, t_pde)
        initial_loss_val = self.initial_loss(x_initial, t_initial)
        boundary_loss_val = self.boundary_loss(t_boundary)

        if self.use_relobralo:
            current_losses = [self.latest_pde_loss, self.latest_initial_loss, self.latest_boundary_loss]
            self.relobralo.update(current_losses)
            weights = self.relobralo.weights.detach()
            return (
                weights[0] * pde_loss_val +
                weights[1] * initial_loss_val +
                weights[2] * boundary_loss_val
            )
        else:
            return pde_loss_val + initial_loss_val + boundary_loss_val

    # Compute final total loss (for evaluation)
    def final_total_loss(self, pde_data, initial_data, boundary_data) -> tuple[float, float, float, float]:
        """Calculate final total loss and its components"""
        x_pde, t_pde = pde_data
        x_initial, t_initial = initial_data
        t_boundary = boundary_data

        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde, t_pde).item()
            initial_loss_val = self.initial_loss(x_initial, t_initial).item()
            boundary_loss_val = self.boundary_loss(t_boundary).item()

            if self.use_relobralo:
                weights = self.relobralo.weights.detach().cpu().numpy()
                total_loss_val = (
                    weights[0] * pde_loss_val +
                    weights[1] * initial_loss_val +
                    weights[2] * boundary_loss_val
                )
            else:
                total_loss_val = pde_loss_val + initial_loss_val + boundary_loss_val

        return total_loss_val, pde_loss_val, initial_loss_val, boundary_loss_val

# Data generation function
def generate_data(
    n_pde: int,
    n_initial: int,
    n_boundary: int,
    n_test: int
) -> tuple:
    # PDE collocation points: x∈[-1,1], t∈[0,1]
    x_pde = (torch.rand(n_pde, 1) * 2 - 1).detach().requires_grad_(True)
    t_pde = torch.rand(n_pde, 1).detach().requires_grad_(True)
    pde_data = (x_pde, t_pde)

    # Initial condition collocation points: x∈[-1,1], t=0
    x_initial = (torch.rand(n_initial, 1) * 2 - 1).detach().requires_grad_(True)
    t_initial = torch.zeros(n_initial, 1).detach().requires_grad_(True)
    initial_data = (x_initial, t_initial)

    # Boundary condition collocation points: t∈[0,1]
    t_boundary = torch.rand(n_boundary, 1).detach().requires_grad_(True)
    boundary_data = t_boundary

    # Test data: grid over x∈[-1,1], t∈[0,1] (no gradients required)
    x_test = torch.linspace(SPACE_DOMAIN[0], SPACE_DOMAIN[1], n_test).reshape(-1, 1)
    t_test = torch.linspace(TIME_DOMAIN[0], TIME_DOMAIN[1], n_test).reshape(-1, 1)
    x_test_grid, t_test_grid = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    x_test_flat = x_test_grid.reshape(-1, 1)
    t_test_flat = t_test_grid.reshape(-1, 1)
    test_data = (x_test_flat, t_test_flat)

    return pde_data, initial_data, boundary_data, test_data

# Training function
def train_with_optimizer(
    model: CGMPINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    step_offset: int = 0,
    global_total_steps: int = None
) -> int:
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    model.train()

    if global_total_steps is None:
        global_total_steps = epochs

    for epoch in range(epochs):
        global_step = step_offset + epoch + 1
        model.set_training_progress(global_step, global_total_steps)

        if model.use_curriculum and epoch % model.curriculum_weight.update_interval == 0:
            model.update_curriculum_weights(x_pde, t_pde)
            print(f"  → Epoch {epoch+1} (Global step {global_step}), curriculum weights updated (tau={model.current_tau:.3f})")

        total_loss = model.compute_weighted_total_loss(pde_data, initial_data, boundary_data)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss_history.append(total_loss.item())

        if (epoch + 1) % 1000 == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            relobralo_info = ""

            pde_loss = model.latest_pde_loss if model.latest_pde_loss is not None else 0.0
            initial_loss = model.latest_initial_loss if model.latest_initial_loss is not None else 0.0
            boundary_loss = model.latest_boundary_loss if model.latest_boundary_loss is not None else 0.0

            if model.use_relobralo:
                weights = model.relobralo.weights.detach().cpu().numpy()
                relobralo_info = f" | ReLoBRaLo Weights(PDE/IC/BC): [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
            print(
                f'Epoch {epoch+1:4d}{curr_info}{relobralo_info} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss:.2e} | Initial Loss: {initial_loss:.2e} | '
                f'Boundary Loss: {boundary_loss:.2e}'
            )

    return step_offset + epochs

# L-BFGS training function
def train_with_lbfgs(
    model: CGMPINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    lr: float,
    max_iter: int,
    step_offset: int = 0,
    global_total_steps: int = None
) -> int:
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    model.train()

    if global_total_steps is None:
        global_total_steps = max_iter

    pde_loss_val = initial_loss_val = boundary_loss_val = 0.0

    if model.use_curriculum:
        model.update_curriculum_weights(x_pde, t_pde)

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, initial_loss_val, boundary_loss_val
        optimizer.zero_grad()
        total_loss = model.compute_weighted_total_loss(pde_data, initial_data, boundary_data)
        total_loss.backward()

        pde_loss_val = model.latest_pde_loss if model.latest_pde_loss is not None else 0.0
        initial_loss_val = model.latest_initial_loss if model.latest_initial_loss is not None else 0.0
        boundary_loss_val = model.latest_boundary_loss if model.latest_boundary_loss is not None else 0.0

        return total_loss

    optimizer = optim.LBFGS(
        model.parameters(), max_iter=1, max_eval=10,
        line_search_fn='strong_wolfe', lr=lr
    )

    for iter_idx in range(max_iter):
        global_step = step_offset + iter_idx + 1
        model.set_training_progress(global_step, global_total_steps)

        total_loss = optimizer.step(closure)

        if model.use_curriculum and (iter_idx % model.curriculum_weight.update_interval == 0):
            model.update_curriculum_weights(x_pde, t_pde)
            print(f"  → Iteration {iter_idx+1} (Global step {global_step}), curriculum weights updated (tau={model.current_tau:.3f})")

        loss_history.append(total_loss.item())

        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            relobralo_info = ""
            if model.use_relobralo:
                weights = model.relobralo.weights.detach().cpu().numpy()
                relobralo_info = f" | ReLoBRaLo Weights(PDE/IC/BC): [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
            print(
                f'Iteration {iter_idx+1}/{max_iter}{curr_info}{relobralo_info} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss_val:.2e} | Initial Loss: {initial_loss_val:.2e} | '
                f'Boundary Loss: {boundary_loss_val:.2e}'
            )

    return step_offset + max_iter

# Two-stage training function
def train_adam_lbfgs(
    model: CGMPINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    global_total_steps = adam_epochs + lbfgs_max_iter

    print(f"\n=== Stage 1: Adam Global Exploration (starting with easy samples) ===")
    print(f"    Global step range: 1 ~ {adam_epochs}, tau: 0 → {adam_epochs/global_total_steps:.3f}")

    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    current_step = train_with_optimizer(
        model=model,
        optimizer=optimizer_adam,
        epochs=adam_epochs,
        pde_data=pde_data,
        initial_data=initial_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        step_offset=0,
        global_total_steps=global_total_steps
    )

    print(f"\n=== Stage 2: L-BFGS Local Refinement (gradually focusing on hard samples) ===")
    print(f"    Global step range: {current_step+1} ~ {global_total_steps}, tau: {current_step/global_total_steps:.3f} → 1.0")

    train_with_lbfgs(
        model=model,
        pde_data=pde_data,
        initial_data=initial_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        lr=lr_lbfgs,
        max_iter=lbfgs_max_iter,
        step_offset=current_step,
        global_total_steps=global_total_steps
    )

# Evaluation function
def evaluate_model(
    model: CGMPINN,
    test_data: tuple,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    x_test, t_test = test_data
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test, t_test)
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        u_exact_l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()
        l2_relative_error = l2_error / u_exact_l2_norm if u_exact_l2_norm > 1e-10 else float('inf')
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        pointwise_error = torch.abs(u_pred - u_exact).numpy()
        u_pred_np = u_pred.numpy()
    return l2_error, l2_relative_error, linf_error, pointwise_error, u_pred_np

# Visualization functions
def plot_loss_curves_by_activation(activation_loss_histories: dict, optim_labels: list) -> None:
    for activation_name, loss_histories in activation_loss_histories.items():
        plt.figure(figsize=(10, 6))
        for loss_history, label in zip(loss_histories, optim_labels):
            plt.plot(loss_history, label=label, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss (Log Scale)', fontsize=12)
        plt.yscale('log')
        plt.title(f'Training Loss Comparison (Activation: {activation_name})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_solutions_by_activation_optimizer(
    activation_results: dict,
    test_data: tuple,
    u_exact_np: np.ndarray,
    optim_labels: list
) -> None:
    x_test, t_test = test_data
    n_test = int(np.sqrt(len(x_test)))

    for activation_name, results in activation_results.items():
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        fig.suptitle(f'Solution Comparison - Activation: {activation_name}', fontsize=24, y=0.99)

        for idx, optim_label in enumerate(optim_labels):
            u_pred = results[optim_label]['u_pred']
            u_pred_grid = u_pred.reshape(n_test, n_test)
            u_exact_grid = u_exact_np.reshape(n_test, n_test)
            error_grid = np.abs(u_pred_grid - u_exact_grid)
            x_grid = x_test.reshape(n_test, n_test)
            t_grid = t_test.reshape(n_test, n_test)

            ax1 = axes[idx, 0]
            im1 = ax1.pcolormesh(t_grid, x_grid, u_pred_grid, cmap=cm.jet, shading='gouraud')
            ax1.set_xlabel('Time (t)', fontsize=10)
            ax1.set_ylabel('Position (x)', fontsize=10)
            ax1.set_title(f'{optim_label} - Predicted', fontsize=12)
            plt.colorbar(im1, ax=ax1, shrink=0.6, aspect=5)

            ax2 = axes[idx, 1]
            im2 = ax2.pcolormesh(t_grid, x_grid, u_exact_grid, cmap=cm.jet, shading='gouraud')
            ax2.set_title('Analytical Solution', fontsize=12)
            plt.colorbar(im2, ax=ax2, shrink=0.6, aspect=5)
            ax2.set_xlabel('Time (t)', fontsize=10)
            ax2.set_ylabel('Position (x)', fontsize=10)

            ax3 = axes[idx, 2]
            im3 = ax3.pcolormesh(t_grid, x_grid, error_grid, cmap=cm.viridis, shading='gouraud')
            ax3.set_xlabel('Time (t)', fontsize=10)
            ax3.set_ylabel('Position (x)', fontsize=10)
            ax3.set_title(f'{optim_label} - Absolute Error', fontsize=12)
            plt.colorbar(im3, ax=ax3, shrink=0.6, aspect=5)

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

# Main function
def train_1D_advection_diffusion_CGMPINN():
    # Hyperparameter settings
    n_layers = 4
    input_dim = 2
    output_dim = 1
    hidden_dim = 50
    n_pde = 3000
    n_initial = 300
    n_boundary = 300
    n_test = 100
    lr_lbfgs_config = {'Tanh': 1.0}
    lr_adam_lbfgs = 0.8
    epochs_sgd_adam = 15000
    max_iter_lbfgs = 15000
    adam_lbfgs_adam_epochs = 5000
    adam_lbfgs_lbfgs_iter = 10000

    # Curriculum Learning + GMM parameters
    gmm_kwargs = {
        'n_components': 4,
        'update_interval': 200,
        'epsilon': 1e-6,
        'beta': 1.0,
        'tau_saturation': 0.8,
        'use_variance_factor': True
    }

    # ReLoBRaLo adaptive weight parameters
    relobralo_kwargs = {
        'n_losses': 3,
        'alpha': 0.999,
        'temperature': 1.0,
        'rho': 0.99,
        'device': 'cpu'
    }

    # Activation functions
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (CGMPINN)', 'L-BFGS (CGMPINN)', 'Adam→L-BFGS (CGMPINN)']

    # Generate data
    print("Generating training and test data...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde, n_initial=n_initial, n_boundary=n_boundary, n_test=n_test
    )
    x_test, t_test = test_data

    # Calculate analytical solution
    pi = np.pi
    u_exact = -torch.exp(-NU * pi**2 * t_test) * torch.sin(pi * (x_test - BETA * t_test))
    u_exact_np = u_exact.numpy()

    # Initialize storage
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}

    # Train by activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting Training - Activation: {activation_name} | Strategy: GMM Curriculum Learning + ReLoBRaLo Adaptive Weights")
        print("="*80)

        current_loss_histories = []
        current_results = {}
        current_training_times = []

        # 1. Adam (GMM + Curriculum Learning + ReLoBRaLo)
        print(f"\n--- {activation_name} + Adam (CGMPINN+ReLoBRaLo) ---")
        model_adam = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs
        )
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        start_time_adam = time.time()
        train_with_optimizer(
            model=model_adam, optimizer=optimizer_adam, epochs=epochs_sgd_adam,
            pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_adam
        )
        training_time_adam = time.time() - start_time_adam
        current_training_times.append(training_time_adam)

        # Evaluate Adam
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam.final_total_loss(
            pde_data, initial_data, boundary_data
        )
        current_results['Adam (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam Training Time: {training_time_adam:.2f}s | L2 Error: {l2_err:.2e} | L∞ Error: {linf_err:.2e}")

        # 2. L-BFGS (GMM + Curriculum Learning + ReLoBRaLo)
        print(f"\n--- {activation_name} + L-BFGS (CGMPINN+ReLoBRaLo) ---")
        model_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=False, relobralo_kwargs=relobralo_kwargs
        )
        loss_history_lbfgs = []
        start_time_lbfgs = time.time()
        train_with_lbfgs(
            model=model_lbfgs, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_lbfgs, lr=lr_lbfgs_config[activation_name], max_iter=max_iter_lbfgs
        )
        training_time_lbfgs = time.time() - start_time_lbfgs
        current_training_times.append(training_time_lbfgs)

        # Evaluate L-BFGS
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_lbfgs.final_total_loss(
            pde_data, initial_data, boundary_data
        )
        current_results['L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS Training Time: {training_time_lbfgs:.2f}s | L2 Error: {l2_err:.2e} | L∞ Error: {linf_err:.2e}")

        # 3. Adam→L-BFGS (GMM + Curriculum Learning + ReLoBRaLo)
        print(f"\n--- {activation_name} + Adam→L-BFGS (CGMPINN+ReLoBRaLo) ---")
        model_adam_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs
        )
        loss_history_adam_lbfgs = []
        start_time_adam_lbfgs = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs, lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs, lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        training_time_adam_lbfgs = time.time() - start_time_adam_lbfgs
        current_training_times.append(training_time_adam_lbfgs)

        # Evaluate Adam→L-BFGS
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam_lbfgs.final_total_loss(
            pde_data, initial_data, boundary_data
        )
        current_results['Adam→L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam_lbfgs)

        # Save Tanh+Adam→LBFGS model
        if activation_name == 'Tanh':
            save_path = "1D_advection_diffusion_tanh_adam2lbfgs_cgmpinn.pth"
            complete_save_dict = {
                'model_state_dict': model_adam_lbfgs.state_dict(),
                'loss_history': loss_history_adam_lbfgs,
                'final_total_loss': final_total,
                'training_time': training_time_adam_lbfgs,
                'activation_name': activation_name,
                'hyper_parameters': {
                    'n_layers': n_layers,
                    'hidden_dim': hidden_dim,
                    'adam_epochs': adam_lbfgs_adam_epochs,
                    'lbfgs_max_iter': adam_lbfgs_lbfgs_iter
                }
            }
            torch.save(complete_save_dict, save_path)
            print(f"✅ Tanh+Adam→LBFGS model saved to: {save_path}")

        print(f"Adam→L-BFGS Training Time: {training_time_adam_lbfgs:.2f}s | L2 Error: {l2_err:.2e} | L∞ Error: {linf_err:.2e}")

        # Save results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times

    # Visualization
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_optimizer(activation_results, test_data, u_exact_np, optim_labels)

    # Print comprehensive comparison table
    print("\n" + "="*160)
    print("Advection-Diffusion Equation - GMM Curriculum Learning × ReLoBRaLo Adaptive Weights × Optimizer Comprehensive Comparison")
    print("="*160)
    header = (f"{'Activation':<10} {'Optimizer':<25} {'Train Time(s)':<12} "
              f"{'L2 Error':<12} {'Rel L2 Error':<12} {'L∞ Error':<12} {'Final Total Loss':<12}")
    print(header)
    print("-"*160)
    for activation_name in activation_names:
        for idx, optim_label in enumerate(optim_labels):
            results = activation_results[activation_name][optim_label]
            train_time = activation_training_times[activation_name][idx]
            print(
                f"{activation_name:<10} {optim_label:<25} "
                f"{train_time:<12.2f} {results['l2_error']:<12.2e} "
                f"{results['l2_relative_error']:<12.2e} {results['linf_error']:<12.2e} "
                f"{results['final_total_loss']:<12.2e}"
            )

    # Print best performance
    print("\n" + "="*130)
    print("Advection-Diffusion Equation - Best Performance by Optimizer (Sorted by L2 Error)")
    print("="*130)
    for activation_name in activation_names:
        print(f"\n【{activation_name} - GMM Curriculum Learning + ReLoBRaLo Adaptive Weights】")
        optim_results = []
        for optim_label, results in activation_results[activation_name].items():
            train_time = activation_training_times[activation_name][optim_labels.index(optim_label)]
            optim_results.append({
                'optimizer': optim_label, 'train_time': train_time,
                'l2_error': results['l2_error'], 'linf_error': results['linf_error'],
                'final_loss': results['final_total_loss']
            })
        optim_results_sorted = sorted(optim_results, key=lambda x: x['l2_error'])
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<25} | Train Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_advection_diffusion_CGMPINN()