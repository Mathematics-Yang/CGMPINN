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

torch.manual_seed(1224)
np.random.seed(1224)

# define constants for the 1D heat equation problem
ALPHA1 = 1.0  
ALPHA2 = 2.0  
k = 10.0 

# define activation functions
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# adaptive loss weights base class
class AdaptiveLossWeights:
    """
    Base class for adaptive loss weights (provides foundation for ReLoBRaLo)
    """
    def __init__(self, n_losses=3, device='cpu'):
        super().__init__()
        self.n_losses = n_losses  
        self.device = device      
        self.weights = torch.ones(n_losses, device=device, dtype=torch.float32)  # initialize weights to 1

class ReLoBRaLoWeights(AdaptiveLossWeights):
    """
    ReLoBRaLo: Relative Loss Balancing with Random Lookback
    Adaptive weights based on loss change rates
    Core ideas:
    - Loss terms that decrease slowly should be given higher weights
    - Use random lookback to smooth weight updates
    """
    def __init__(self, n_losses=3, alpha=0.999, temperature=1.0, 
                 rho=0.99, device='cpu'):
        super().__init__(n_losses, device)
        self.alpha = alpha           # exponential moving average coefficient
        self.temperature = temperature  # softmax temperature
        self.rho = rho               # random lookback probability
        
        # store historical losses
        self.loss_history = [[] for _ in range(n_losses)]
        self.ema_losses = None  # exponential moving average of losses
        self.initial_losses = None  # initial losses (for normalization)
        
    def update(self, losses, **kwargs):
        """
        Update weights
        :param losses: Current list of loss values [pde_loss, ic_loss, bc_loss]
        """
        losses_tensor = torch.tensor(losses, device=self.device, dtype=torch.float32)
        
        # record history
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss)
        
        # initialize EMA and initial losses
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.clone()
            self.ema_losses = losses_tensor.clone()
            return
        
        # update EMA losses
        self.ema_losses = self.alpha * self.ema_losses + (1 - self.alpha) * losses_tensor
        
        # calculate relative loss change rates
        # random lookback: use EMA with probability rho, otherwise use current value
        if np.random.rand() < self.rho:
            reference = self.ema_losses
        else:
            # randomly select a point from history
            lookback_idx = np.random.randint(0, max(1, len(self.loss_history[0]) - 1))
            reference = torch.tensor(
                [self.loss_history[i][lookback_idx] for i in range(self.n_losses)],
                device=self.device, dtype=torch.float32
            )
        
        # calculate relative change rates
        relative_losses = losses_tensor / (reference + 1e-8)
        
        # use softmax to calculate weights
        scaled_losses = relative_losses / self.temperature
        self.weights = self.n_losses * torch.softmax(scaled_losses, dim=0)

# GMM weight calculation utility class
class GMMCurriculumWeight:
    def __init__(self, n_components=4, update_interval=200, epsilon=1e-6, 
                 beta=1.0,  # increase beta to enhance distinction
                 tau_saturation=0.8,
                 use_variance_factor=False):  # whether to use variance factor
        """
        Initialize GMM curriculum learning weight calculator (from easy to hard)
        :param n_components: Number of Gaussian components K
        :param update_interval: Weight update interval
        :param epsilon: Small value to prevent division by zero
        :param beta: Intensity coefficient for curriculum learning (larger beta means more drastic difficulty switching)
        :param tau_saturation: Tau saturation point (default 0.8, meaning tau reaches 1 at 80% of the steps)
        :param use_variance_factor: Whether to use variance factor to optimize weights
        """
        self.n_components = n_components
        self.update_interval = update_interval
        self.epsilon = epsilon
        self.beta = beta
        self.tau_saturation = tau_saturation
        self.use_variance_factor = use_variance_factor
        self.gmm = GaussianMixture(n_components=n_components, random_state=1224)
        self._last_valid_weights = None  

    def compute_weights(self, residuals: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Calculate weights based on residuals and training progress 
        :param residuals: PDE residual tensor (n_pde, 1)
        :param tau: Training progress (0=early, 1=late)
        :return: Weight tensor for each collocation point (n_pde, 1)
        """
        res_np = residuals.detach().cpu().numpy().reshape(-1, 1)
        
        try:
            # 1. GMM fit residual distribution  
            self.gmm.fit(res_np)
            gamma = self.gmm.predict_proba(res_np)  # (n_pde, K) posterior probabilities
            sigma_sq = self.gmm.covariances_.flatten()  # (K,) variances of each component
            
            # 2. Use posterior probabilities to weight component difficulty
            res_squared = res_np.flatten() ** 2
            component_difficulty = np.array([
                np.sum(gamma[:, j] * res_squared) / (np.sum(gamma[:, j]) + self.epsilon)
                for j in range(self.n_components)
            ])
            
            # 3. Normalize difficulty to [0, 1] (handle extreme cases to prevent division by zero)
            diff_min, diff_max = component_difficulty.min(), component_difficulty.max()
            if diff_max - diff_min > self.epsilon:
                normalized_diff = (component_difficulty - diff_min) / (diff_max - diff_min)
            else:
                normalized_diff = np.zeros_like(component_difficulty)
            
            # 4. Curriculum learning weights (larger beta increases distinction)
            easy_weight = np.exp(-self.beta * normalized_diff)  # Easy sample weight: lower difficulty → higher weight
            hard_weight = np.exp(-self.beta * (1 - normalized_diff))  # Hard sample weight: higher difficulty → higher weight
            
            # 5. Smooth transition controlled by tau (from easy to hard)
            curriculum_weight = (1 - tau) * easy_weight + tau * hard_weight
            
            # 6. Optional variance factor (improves stability early in training, diminishes with increasing tau)
            if self.use_variance_factor:
                variance_factor = 1 / (sigma_sq + self.epsilon)
                variance_factor = variance_factor / variance_factor.max()  # Normalize variance factor
                effective_variance_factor = (1 - tau) * variance_factor + tau * 1.0  # Smoothly reduce variance influence
                component_weights = curriculum_weight * effective_variance_factor
            else:
                component_weights = curriculum_weight
            
            # 7. Sample weights = weighted sum of posterior probabilities × component weights
            point_weights = np.sum(gamma * component_weights[np.newaxis, :], axis=1)
            self._last_valid_weights = point_weights.copy()  # Update valid weights cache
        
        except Exception as e:
            print(f"GMM training failed: {e}")
            # Exception handling: use last valid weights or all ones
            if self._last_valid_weights is not None and len(self._last_valid_weights) == len(res_np):
                point_weights = self._last_valid_weights
            else:
                point_weights = np.ones(len(res_np))
        
        # 8. Normalize weights (ensure mean is 1, preserving overall loss scale)
        point_weights = point_weights / (point_weights.mean() + self.epsilon)
        
        # Convert to torch tensor and return (matching input device)
        return torch.tensor(point_weights.reshape(-1, 1), dtype=torch.float32).to(residuals.device)

# CGMPINN model class
class CGMPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, 
                 use_curriculum=True, gmm_kwargs=None,
                 use_relobralo=True, relobralo_kwargs=None):  
        super(CGMPINN, self).__init__()
        self.activation = activation
        self.use_curriculum = use_curriculum  # Curriculum learning switch
        self.use_relobralo = use_relobralo    # ReLoBRaLo adaptive weight switch
        self.total_train_steps = 0  # Total training steps (used to calculate tau)
        self.current_tau = 0.0     # Current training progress

        # Build network architecture
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
        
        # Cache latest component loss values
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
        if not self.use_curriculum:
            return
        u_t, u_xx = self.compute_gradients(x, t)
        f = self.source_term(x, t)
        pde_residual = u_t - u_xx - f
        self.sample_weights = self.curriculum_weight.compute_weights(pde_residual, self.current_tau)

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        if not self.use_curriculum:
            return
        if total_steps == 0:
            self.current_tau = 0.0
        else:
            # Adapt GMM's tau saturation point (default tau reaches 1 at 80% of total steps)
            tau_base = total_steps * self.curriculum_weight.tau_saturation
            self.current_tau = min(current_step / tau_base, 1.0)  # Limit tau ≤ 1

    # Forward pass
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)
        return self.layers(input_tensor)
    
    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.forward(x, t)
        u_t = torch.autograd.grad(
            outputs=u, inputs=t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        u_x = torch.autograd.grad(
            outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        return u_t, u_xx
    
    # Define source term f(x,t)
    def source_term(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        sin_alpha1pi_x = torch.sin(ALPHA1 * np.pi * x)
        tanh_kx = torch.tanh(k * x)
        sin_alpha2pi_t = torch.sin(ALPHA2 * np.pi * t)
        cos_alpha2pi_t = torch.cos(ALPHA2 * np.pi * t)
        
        # u = (sin(α1πx) + tanh(kx)) * sin(α2πt)
        # u_t = (sin(α1πx) + tanh(kx)) * α2π cos(α2πt)
        u_t_exact = (sin_alpha1pi_x + tanh_kx) * ALPHA2 * np.pi * cos_alpha2pi_t
        
        # u_x = (α1π cos(α1πx) + k sech²(kx)) * sin(α2πt)
        sech_kx_sq = 1 / (torch.cosh(k * x) ** 2)
        u_x_exact = (ALPHA1 * np.pi * torch.cos(ALPHA1 * np.pi * x) + k * sech_kx_sq) * sin_alpha2pi_t
        
        # u_xx = [ - α1²π² sin(α1πx) - 2k² sech²(kx) tanh(kx) ] * sin(α2πt)
        u_xx_exact = (
            - (ALPHA1 ** 2) * (np.pi ** 2) * sin_alpha1pi_x 
            - 2 * (k ** 2) * sech_kx_sq * tanh_kx
        ) * sin_alpha2pi_t
        
        # f(x,t) = u_t - u_xx
        f = u_t_exact - u_xx_exact
        return f

    def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """PDE 损失：u_t - u_xx - f(x,t) = 0"""
        u_t, u_xx = self.compute_gradients(x, t)
        f = self.source_term(x, t)
        pde_residual = u_t - u_xx - f  

        if self.use_curriculum and self.sample_weights is not None:
            # Weighted loss: higher weights for simpler samples early on, gradually decreasing
            weighted_residual = self.sample_weights.detach() * (pde_residual ** 2)
            pde_loss_val = torch.mean(weighted_residual)
        else:
            pde_loss_val = torch.mean(pde_residual ** 2)
        
        # Cache PDE loss value
        self.latest_pde_loss = pde_loss_val.item()
        
        return pde_loss_val
    
    def initial_loss(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u_exact = (torch.sin(ALPHA1 * np.pi * x) + torch.tanh(k * x)) * torch.sin(ALPHA2 * np.pi * t0)
        initial_residual = u_pred - u_exact
        initial_loss_val = torch.mean(initial_residual ** 2)
        self.latest_initial_loss = initial_loss_val.item()
        
        return initial_loss_val
    
    def boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        x0 = torch.zeros_like(t)
        u0_pred = self.forward(x0, t)
        u0_exact = (torch.sin(ALPHA1 * np.pi * x0) + torch.tanh(k * x0)) * torch.sin(ALPHA2 * np.pi * t)
        x1 = torch.ones_like(t)
        u1_pred = self.forward(x1, t)
        u1_exact = (torch.sin(ALPHA1 * np.pi * x1) + torch.tanh(k * x1)) * torch.sin(ALPHA2 * np.pi * t)
        boundary_residual1 = u0_pred - u0_exact
        boundary_residual2 = u1_pred - u1_exact
        boundary_loss_val = torch.mean(boundary_residual1 ** 2) + torch.mean(boundary_residual2 ** 2)
        self.latest_boundary_loss = boundary_loss_val.item()
        
        return boundary_loss_val
    
    def compute_weighted_total_loss(self, pde_data, initial_data, boundary_data) -> torch.Tensor:
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
    
    def final_total_loss(self, pde_data, initial_data, boundary_data) -> tuple[float, float, float, float]:
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
    x_pde = torch.rand(n_pde, 1, requires_grad=True)
    t_pde = torch.rand(n_pde, 1, requires_grad=True)
    pde_data = (x_pde, t_pde)
    
    x_initial = torch.rand(n_initial, 1, requires_grad=True)
    t_initial = torch.zeros_like(x_initial, requires_grad=True)
    initial_data = (x_initial, t_initial)
    
    t_boundary = torch.rand(n_boundary, 1, requires_grad=True)
    boundary_data = t_boundary
    
    x_test = torch.linspace(0, 1, n_test).reshape(-1, 1)
    t_test = torch.linspace(0, 1, n_test).reshape(-1, 1)
    x_test_grid, t_test_grid = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    x_test_flat = x_test_grid.reshape(-1, 1)
    t_test_flat = t_test_grid.reshape(-1, 1)
    test_data = (x_test_flat, t_test_flat)
    
    return pde_data, initial_data, boundary_data, test_data

# Training function with optimizers (Adam)
def train_with_optimizer(
    model: CGMPINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    step_offset: int = 0,           # Step offset
    global_total_steps: int = None  # Global total steps
) -> int:  # Return cumulative steps
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    model.train()

    # If global total steps not specified, use current phase's epochs
    if global_total_steps is None:
        global_total_steps = epochs

    for epoch in range(epochs):
        # Calculate global step (offset + current epoch)
        global_step = step_offset + epoch + 1
        
        # Use global step to calculate tau
        model.set_training_progress(global_step, global_total_steps)
        
        # Periodically update Gaussian mixture model weights
        if model.use_curriculum and epoch % model.curriculum_weight.update_interval == 0:
            model.update_curriculum_weights(x_pde, t_pde)
            print(f"  → Epoch {epoch+1} (Global step {global_step}), curriculum weights (tau={model.current_tau:.3f})")

        # Loss calculation and backpropagation (using ReLoBRaLo weighted total loss)
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
                relobralo_info = f" | ReLoBRaLo weights (PDE/IC/BC): [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
            print(
                f'Epoch {epoch+1:4d}{curr_info}{relobralo_info} | Total loss: {total_loss.item():.2e} | '
                f'PDE loss: {pde_loss:.2e} | Initial condition loss: {initial_loss:.2e} | '
                f'Boundary condition loss: {boundary_loss:.2e}'
            )
    
    # Return cumulative global steps
    return step_offset + epochs

# L-BFGS training function (adapted for ReLoBRaLo and curriculum learning)
def train_with_lbfgs(
    model: CGMPINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    lr: float,
    max_iter: int,
    step_offset: int = 0,           # Step offset
    global_total_steps: int = None  # Global total steps
) -> int:  # Return cumulative steps
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    model.train()

    # If global total steps not specified, use current phase's max_iter
    if global_total_steps is None:
        global_total_steps = max_iter

    pde_loss_val = initial_loss_val = boundary_loss_val = 0.0

    # Initialize curriculum learning weights
    if model.use_curriculum:
        model.update_curriculum_weights(x_pde, t_pde)

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, initial_loss_val, boundary_loss_val
        optimizer.zero_grad()
        # Use ReLoBRaLo weighted total loss
        total_loss = model.compute_weighted_total_loss(pde_data, initial_data, boundary_data)
        total_loss.backward()
        
        # Cache component loss values (for printing)
        pde_loss_val = model.latest_pde_loss if model.latest_pde_loss is not None else 0.0
        initial_loss_val = model.latest_initial_loss if model.latest_initial_loss is not None else 0.0
        boundary_loss_val = model.latest_boundary_loss if model.latest_boundary_loss is not None else 0.0
        
        return total_loss

    optimizer = optim.LBFGS(
        model.parameters(), max_iter=1, max_eval=10,
        line_search_fn='strong_wolfe', lr=lr
    )

    for iter_idx in range(max_iter):
        # Calculate global step
        global_step = step_offset + iter_idx + 1
        
        # Use global step to calculate tau
        model.set_training_progress(global_step, global_total_steps)
        
        total_loss = optimizer.step(closure)

        # Periodically update curriculum learning weights
        if model.use_curriculum and (iter_idx % model.curriculum_weight.update_interval == 0):
            model.update_curriculum_weights(x_pde, t_pde)
            print(f"  → Iteration {iter_idx+1} (global step {global_step}), curriculum weights updated (tau={model.current_tau:.3f})")

        loss_history.append(total_loss.item())

        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            relobralo_info = ""
            if model.use_relobralo:
                weights = model.relobralo.weights.detach().cpu().numpy()
                relobralo_info = f" | ReLoBRaLo weights (PDE/IC/BC): [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
            print(
                f'Iteration {iter_idx+1}/{max_iter}{curr_info}{relobralo_info} | Total loss: {total_loss.item():.2e} | '
                f'PDE loss: {pde_loss_val:.2e} | Initial condition loss: {initial_loss_val:.2e} | '
                f'Boundary condition loss: {boundary_loss_val:.2e}'
            )
    
    # Return cumulative global steps
    return step_offset + max_iter

# Two-phase training function (Adam→L-BFGS, adapted for ReLoBRaLo and curriculum learning)
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
    # Calculate global total steps (Adam + L-BFGS)
    global_total_steps = adam_epochs + lbfgs_max_iter
    
    print(f"\n=== Phase 1: Adam Global Exploration (Starting with simple samples) ===")
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
        step_offset=0,                        # Adam starts from 0
        global_total_steps=global_total_steps # Use global total steps
    )
    
    print(f"\n=== Phase 2: L-BFGS Local Fine-tuning (Gradually focusing on difficult samples) ===")
    print(f"    Global step range: {current_step+1} ~ {global_total_steps}, tau: {current_step/global_total_steps:.3f} → 1.0")
    
    train_with_lbfgs(
        model=model, 
        pde_data=pde_data, 
        initial_data=initial_data, 
        boundary_data=boundary_data,
        loss_history=loss_history, 
        lr=lr_lbfgs, 
        max_iter=lbfgs_max_iter,
        step_offset=current_step,             # L-BFGS continues from where Adam ended
        global_total_steps=global_total_steps # Use global total steps
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

# Visualization function
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

# main training function
def train_1D_heat_CGMPINN_with_activations_and_source_term():
    # Hyperparameter settings
    n_layers = 4
    input_dim = 2
    output_dim = 1
    hidden_dim = 50
    n_pde = 1500
    n_initial = 300
    n_boundary = 300
    n_test = 100  
    lr_lbfgs_config = {'Tanh': 1.0}
    lr_adam_lbfgs = 0.8
    epochs_sgd_adam = 15000 
    max_iter_lbfgs = 15000  
    adam_lbfgs_adam_epochs = 5000
    adam_lbfgs_lbfgs_iter = 10000

    # Curriculum learning + GMM parameters
    gmm_kwargs = {
        'n_components': 4, # Using 4 Gaussian components
        'update_interval': 200,  # Update weights every 200 steps
        'epsilon': 1e-6,
        'beta': 1.0,  # Increase beta to enhance difficulty differentiation
        'tau_saturation': 0.8,  # Tau reaches saturation at 80% of steps
        'use_variance_factor': True  # Enable variance factor to improve early training stability
    }
    
    # ReLoBRaLo adaptive weight parameters (adjustable as needed)
    relobralo_kwargs = {
        'n_losses': 3,        # 3 loss components: PDE/Initial/Boundary
        'alpha': 0.999,       # Exponential moving average coefficient
        'temperature': 1.0,   # Softmax temperature (smaller means more distinct weights)
        'rho': 0.99,          # Random rollback probability
        'device': 'cpu'       # Device (change to 'cuda' if GPU is available)
    }

    # Activation functions
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (CGMPINN)', 'L-BFGS (CGMPINN)', 'Adam→L-BFGS (CGMPINN)']

    # Generate data
    print("Generating training and testing data...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde, n_initial=n_initial, n_boundary=n_boundary, n_test=n_test
    )
    x_test, t_test = test_data

    # Calculate exact solution u(x,t) = (sin(α1πx) + tanh(kx)) * sin(α2πt)
    u_exact = (torch.sin(ALPHA1 * np.pi * x_test) + torch.tanh(k * x_test)) * torch.sin(ALPHA2 * np.pi * t_test)
    u_exact_np = u_exact.numpy()

    # Initialize storage
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}

    # Train by activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name} | Strategy: GMM Curriculum Learning + ReLoBRaLo Adaptive Weights")
        print("="*80)

        current_loss_histories = []
        current_results = {}
        current_training_times = []

        # 1. Adam (GMM + Curriculum Learning + ReLoBRaLo)
        print(f"\n--- {activation_name} + Adam (CGMPINN+ReLoBRaLo) ---")
        model_adam = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs  # ReLoBRaLo can be optionally enabled
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
        print(f"Adam training time: {training_time_adam:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")

        # 2. L-BFGS (GMM + Curriculum Learning + ReLoBRaLo)
        print(f"\n--- {activation_name} + L-BFGS (CGMPINN+ReLoBRaLo) ---")
        model_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=False, relobralo_kwargs=relobralo_kwargs  # Since L-BFGS is a second-order method, ReLoBRaLo is not enabled
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
        print(f"L-BFGS training time: {training_time_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")

        # 3. Adam→L-BFGS (GMM + Curriculum Learning + ReLoBRaLo)
        print(f"\n--- {activation_name} + Adam→L-BFGS (CGMPINN+ReLoBRaLo) ---")
        model_adam_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs  # ReLoBRaLo can be optionally enabled
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
        if activation_name == 'Tanh':  
            # Define save path
            save_path = "1D_heat_tanh_adam2lbfgs_cgmpinn.pth"
    
            # Pack all contents to be saved
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
    
            # Save custom dictionary
            torch.save(complete_save_dict, save_path)
            print(f"✅ Tanh+Adam→LBFGS complete information saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {training_time_adam_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")

        # save current activation function results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times

    # Visualize loss curves and solution comparisons
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_optimizer(activation_results, test_data, u_exact_np, optim_labels)

    # Print comprehensive comparison table
    print("\n" + "="*160)
    print("All Activation Functions × GMM Curriculum Learning × ReLoBRaLo Adaptive Weights × Optimizers Comprehensive Comparison Table")
    print("="*160)
    header = (f"{'Activation':<10} {'Optimizer':<25} {'Training Time(s)':<12} "
              f"{'Standard L2 Error':<12} {'L2 Relative Error':<12} {'L∞ Error':<12} {'Final Total Loss':<12}")
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

    # Detailed ranking by activation function
    print("\n" + "="*130)
    print("Best performances of each activation function (GMM Curriculum Learning + ReLoBRaLo Adaptive Weights) sorted by Standard L2 Error")
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
                f"  Rank {i}: {res['optimizer']:<25} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_heat_CGMPINN_with_activations_and_source_term()