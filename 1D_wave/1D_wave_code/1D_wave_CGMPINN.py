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

# define activation functions
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# define equation parameters
ALPHA1 = 1.0    
ALPHA2 = 1.0    
GAMMA = 0.1     
BETA = GAMMA   
c = np.sqrt(GAMMA**2 + (ALPHA2 * np.pi)**2) / (ALPHA1 * np.pi)

# adaptive loss weights base class
class AdaptiveLossWeights:
    """
    Base class for adaptive loss weights (foundation for ReLoBRaLo)
    """
    def __init__(self, n_losses=3, device='cpu'):
        super().__init__()
        self.n_losses = n_losses  # number of loss components
        self.device = device      # computation device
        self.weights = torch.ones(n_losses, device=device, dtype=torch.float32)  # initialize weights to 1

class ReLoBRaLoWeights(AdaptiveLossWeights):
    """
    ReLoBRaLo: Relative Loss Balancing with Random Lookback
    Adaptive weights based on loss change rates
    """
    def __init__(self, n_losses=3, alpha=0.999, temperature=1.0, 
                 rho=0.99, device='cpu'):
        super().__init__(n_losses, device)
        self.alpha = alpha           # exponential moving average coefficient
        self.temperature = temperature  # softmax temperature
        self.rho = rho               # random lookback probability
        
        # store historical losses
        self.loss_history = [[] for _ in range(n_losses)]
        self.ema_losses = None  # exponential moving average losses
        self.initial_losses = None  # initial losses (for normalization)
        
    def update(self, losses, **kwargs):
        """
        Update weights
        :param losses: current list of loss values [pde_loss, initial_loss, boundary_loss]
        """
        losses_tensor = torch.tensor(losses, device=self.device, dtype=torch.float32)
        
        # record history
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss)
        
        # initialize
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.clone()
            self.ema_losses = losses_tensor.clone()
            return
        
        # update EMA losses
        self.ema_losses = self.alpha * self.ema_losses + (1 - self.alpha) * losses_tensor
        
        # compute relative loss change rates
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
        
        # compute relative change rates
        relative_losses = losses_tensor / (reference + 1e-8)
        
        # compute weights using softmax
        scaled_losses = relative_losses / self.temperature
        self.weights = self.n_losses * torch.softmax(scaled_losses, dim=0)

# GMM weight calculation utility class
class GMMCurriculumWeight:
    def __init__(self, n_components=4, update_interval=200, epsilon=1e-6, 
                 beta=1.0, tau_saturation=0.8, use_variance_factor=False):
        """
        Initialize GMM curriculum learning weight calculator (from easy to hard)
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
        Compute weights based on residuals and training progress (core: tau controls the transition from easy to hard)
        :param residuals: PDE residual tensor (n_pde, 1)
        :param tau: training progress (0=early, 1=late)
        :return: weight tensor for each collocation point (n_pde, 1)
        """
        res_np = residuals.detach().cpu().numpy().reshape(-1, 1)
        
        try:
            # 1. GMM fit residual distribution
            self.gmm.fit(res_np)
            gamma = self.gmm.predict_proba(res_np)  # (n_pde, K) posterior probabilities
            sigma_sq = self.gmm.covariances_.flatten()  # (K,) variances of components
            
            # 2. Compute component difficulty weighted by posterior probabilities
            res_squared = res_np.flatten() ** 2
            component_difficulty = np.array([
                np.sum(gamma[:, j] * res_squared) / (np.sum(gamma[:, j]) + self.epsilon)
                for j in range(self.n_components)
            ])
            
            # 3. Normalize difficulty to [0, 1]
            diff_min, diff_max = component_difficulty.min(), component_difficulty.max()
            if diff_max - diff_min > self.epsilon:
                normalized_diff = (component_difficulty - diff_min) / (diff_max - diff_min)
            else:
                normalized_diff = np.zeros_like(component_difficulty)
            
            # 4. Curriculum learning weights
            easy_weight = np.exp(-self.beta * normalized_diff)  # easy sample weights
            hard_weight = np.exp(-self.beta * (1 - normalized_diff))  # hard sample weights
            
            # 5. Smooth transition controlled by tau (from easy to hard)
            curriculum_weight = (1 - tau) * easy_weight + tau * hard_weight
            
            # 6. Optional variance factor
            if self.use_variance_factor:
                variance_factor = 1 / (sigma_sq + self.epsilon)
                variance_factor = variance_factor / variance_factor.max()
                effective_variance_factor = (1 - tau) * variance_factor + tau * 1.0
                component_weights = curriculum_weight * effective_variance_factor
            else:
                component_weights = curriculum_weight
            
            # 7. Sample weights = weighted sum of posterior probabilities × component weights
            point_weights = np.sum(gamma * component_weights[np.newaxis, :], axis=1)
            self._last_valid_weights = point_weights.copy()
        
        except Exception as e:
            print(f"GMM training failed: {e}")
            # Exception handling: use last valid weights or all ones
            if self._last_valid_weights is not None and len(self._last_valid_weights) == len(res_np):
                point_weights = self._last_valid_weights
            else:
                point_weights = np.ones(len(res_np))
        
        # 8. Normalize weights (ensure mean is 1)
        point_weights = point_weights / (point_weights.mean() + self.epsilon)
        
        # Convert to torch tensor and return
        return torch.tensor(point_weights.reshape(-1, 1), dtype=torch.float32).to(residuals.device)

# CGMPINN model
class CGMPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, 
                 use_curriculum=True, gmm_kwargs=None,
                 use_relobralo=True, relobralo_kwargs=None):
        super(CGMPINN, self).__init__()
        self.activation = activation
        self.use_curriculum = use_curriculum  
        self.use_relobralo = use_relobralo    
        self.total_train_steps = 0  # Total training steps
        self.current_tau = 0.0      # Current training progress

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
            model_device = next(self.parameters()).device if self.parameters() else 'cpu'
            relobralo_kwargs['device'] = model_device
            self.relobralo = ReLoBRaLoWeights(**relobralo_kwargs)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)  
        return self.layers(input_tensor)
    
    def update_curriculum_weights(self, x: torch.Tensor, t: torch.Tensor) -> None:
        if not self.use_curriculum:
            return
        u_t, u_tt, _, u_xx = self.compute_gradients(x, t)
        pde_residual = u_tt + 2 * BETA * u_t - (c ** 2) * u_xx
        self.sample_weights = self.curriculum_weight.compute_weights(pde_residual, self.current_tau)

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        if total_steps == 0:
            self.current_tau = 0.0
        else:
            tau_base = total_steps * self.curriculum_weight.tau_saturation if self.use_curriculum else total_steps
            self.current_tau = min(current_step / tau_base, 1.0)  

    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.forward(x, t) 
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        u_tt = torch.autograd.grad(
            outputs=u_t,
            inputs=t,
            grad_outputs=torch.ones_like(u_t),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        return u_t, u_tt, u_x, u_xx

    def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_t, u_tt, _, u_xx = self.compute_gradients(x, t)
        pde_residual = u_tt + 2 * BETA * u_t - (c ** 2) * u_xx

        if self.use_curriculum and self.sample_weights is not None:
            weighted_residual = self.sample_weights.detach() * (pde_residual ** 2)
            pde_loss_val = torch.mean(weighted_residual)
        else:
            pde_loss_val = torch.mean(pde_residual ** 2)
        
        self.latest_pde_loss = pde_loss_val.item()
        
        return pde_loss_val
    
    def initial_loss(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u0_exact = torch.sin(ALPHA1 * np.pi * x)
        initial_residual1 = u_pred - u0_exact
        
        u_t, _, _, _ = self.compute_gradients(x, t0)
        v0_exact = -GAMMA * torch.sin(ALPHA1 * np.pi * x)
        initial_residual2 = u_t - v0_exact
        
        initial_loss_val = torch.mean(initial_residual1 ** 2) + torch.mean(initial_residual2 ** 2)
        
        self.latest_initial_loss = initial_loss_val.item()
        
        return initial_loss_val
    
    def boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        x0 = torch.zeros_like(t)
        u0_pred = self.forward(x0, t)
        u0_exact = torch.exp(-GAMMA * t) * torch.sin(ALPHA1 * np.pi * x0) * torch.cos(ALPHA2 * np.pi * t)

        x1 = torch.ones_like(t)
        u1_pred = self.forward(x1, t)
        u1_exact = torch.exp(-GAMMA * t) * torch.sin(ALPHA1 * np.pi * x1) * torch.cos(ALPHA2 * np.pi * t)
        
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

# generate training and testing data
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

# training function (Adam)
def train_with_optimizer(
    model: CGMPINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    step_offset: int = 0,           # step offset
    global_total_steps: int = None  # total global steps
) -> int:  # return cumulative steps
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    model.train()

    # If total global steps not specified, use current phase's epochs
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

            # Get cached component losses
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

# Training function (L-BFGS)
def train_with_lbfgs(
    model: CGMPINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    lr: float,
    max_iter: int,
    step_offset: int = 0,           # Step offset
    global_total_steps: int = None  # Total global steps
) -> int:  # Return cumulative steps
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    model.train()

    # If total global steps not specified, use current phase's max_iter
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
            print(f"  → Iteration {iter_idx+1} (Global step {global_step}), current curriculum weight (tau={model.current_tau:.3f})")

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

# Two-stage training function (Adam→L-BFGS)
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
    # Calculate total global steps (Adam + L-BFGS)
    global_total_steps = adam_epochs + lbfgs_max_iter
    
    print(f"\n=== Stage 1: Adam Global Exploration (Starting from simple samples) ===")
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
        step_offset=0,                        # Adam starts from step 0
        global_total_steps=global_total_steps # Use total global steps
    )
    
    print(f"\n=== Stage 2: L-BFGS Local Fine-tuning (Gradually focusing on difficult samples) ===")
    print(f"    Global step range: {current_step+1} ~ {global_total_steps}, tau: {current_step/global_total_steps:.3f} → 1.0")
    
    train_with_lbfgs(
        model=model, 
        pde_data=pde_data, 
        initial_data=initial_data, 
        boundary_data=boundary_data,
        loss_history=loss_history, 
        lr=lr_lbfgs, 
        max_iter=lbfgs_max_iter,
        step_offset=current_step,             # L-BFGS continues from where Adam left off
        global_total_steps=global_total_steps # Use total global steps
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

def plot_solutions_by_activation_and_optimizer(
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

def plot_damping_effect(x_grid_flat, t_grid_flat, u_pred_results: dict, u_exact_flat):
    x_grid_flat = x_grid_flat.ravel()  
    t_grid_flat = t_grid_flat.ravel()  
    u_exact_flat = u_exact_flat.ravel()

    n_test = int(np.sqrt(len(x_grid_flat)))
    if n_test * n_test != len(x_grid_flat):
        raise ValueError(f"Number of data points {len(x_grid_flat)} is not a perfect square, please check the test data generation logic")
    
    x_coords = np.unique(x_grid_flat)  
    t_coords = np.unique(t_grid_flat)  
    u_exact_grid = u_exact_flat.reshape(n_test, n_test)
    
    optim_labels = list(u_pred_results.keys())
    n_optims = len(optim_labels)
    fig, axes = plt.subplots(n_optims, 3, figsize=(21, 6*n_optims))
    if n_optims == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Damping Characteristics Comparison by Optimizers', fontsize=18, y=0.98)
    
    for row_idx, optim_label in enumerate(optim_labels):
        u_pred_grid = u_pred_results[optim_label].ravel().reshape(n_test, n_test)
        
        # First column: Spatial waveforms at different time points
        ax1 = axes[row_idx, 0]
        time_indices = [0, n_test//4, n_test//2, 3*n_test//4, n_test-1]
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(time_indices)))
        
        for idx, t_idx in enumerate(time_indices):
            t_val = t_coords[t_idx]
            ax1.plot(x_coords, u_exact_grid[:, t_idx], 
                     color=colors[idx], linewidth=2.5, linestyle='-',
                     label=f'$t={t_val:.2f}$')
            ax1.plot(x_coords, u_pred_grid[:, t_idx], 
                     color=colors[idx], linewidth=2, linestyle='--', marker='o',
                     markevery=10, markersize=4)
        
        ax1.set_xlabel('Position $x$', fontsize=12)
        ax1.set_ylabel('$u(x,t)$', fontsize=12)
        ax1.set_title(f'{optim_label} - Spatial Profiles', fontsize=14)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='k', linewidth=0.5)
        ax1.set_xlim([0, 1])
        
        ax1.annotate('', xy=(0.5, u_exact_grid[n_test//2, 0]), 
                         xytext=(0.5, u_exact_grid[n_test//2, -1]),
                         arrowprops=dict(arrowstyle='<->', color='green', lw=2))
        ax1.text(0.55, 0.5, 'Amplitude\nDecay', fontsize=10, color='green')
        
        # Second column: Temporal evolution at x=0.5
        ax2 = axes[row_idx, 1]
        x_mid_idx = np.argmin(np.abs(x_coords - 0.5))
        t_vals = t_coords 
        
        ax2.plot(t_vals, u_exact_grid[x_mid_idx, :], 
                 'b-', linewidth=2.5, label='Analytical Solution')
        ax2.plot(t_vals, u_pred_grid[x_mid_idx, :], 
                 'r--', linewidth=2, label=f'{optim_label} Prediction')
        
        # Plot decay envelope
        envelope = np.exp(-GAMMA * t_vals)
        ax2.plot(t_vals, envelope, 'g-.', linewidth=2, label=f'Envelope: $e^{{-\\gamma t}}$')
        ax2.plot(t_vals, -envelope, 'g-.', linewidth=2)
        ax2.fill_between(t_vals, -envelope, envelope, alpha=0.1, color='green')
        
        ax2.set_xlabel('Time $t$', fontsize=12)
        ax2.set_ylabel('$u(x=0.5, t)$', fontsize=12)
        ax2.set_title(f'{optim_label} - Temporal Evolution', fontsize=14)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axhline(y=0, color='k', linewidth=0.5)
        ax2.set_xlim([0, 1])
        
        # Third column: Error analysis at x=0.5
        ax3 = axes[row_idx, 2]
        u_exact_mid = u_exact_grid[x_mid_idx, :]
        u_pred_mid = u_pred_grid[x_mid_idx, :]
        abs_error = np.abs(u_pred_mid - u_exact_mid)
        rel_error = abs_error / (np.abs(u_exact_mid) + 1e-8) * 100  
        
        ax3.plot(t_vals, abs_error, 'darkred', linewidth=2.5, label='Absolute Error')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(t_vals, rel_error, 'orange', linewidth=2, linestyle='--', label='Relative Error (%)')
        ax3_twin.set_ylabel('Relative Error (%)', fontsize=12, color='orange')
        ax3_twin.tick_params(axis='y', labelcolor='orange')
                
        ax3.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.8)
        ax3.fill_between(t_vals, 0, abs_error, alpha=0.2, color='darkred')
        
        mean_error = np.mean(abs_error)
        max_error = np.max(abs_error)
        rmse = np.sqrt(np.mean(abs_error**2))
        error_text = (f'Mean Error: {mean_error:.4f}\n'
                      f'Max Error: {max_error:.4f}\n'
                      f'RMSE: {rmse:.4f}')
        ax3.text(0.05, 0.95, error_text, transform=ax3.transAxes,
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        handles1, labels1 = ax3.get_legend_handles_labels()
        handles2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=9)

        ax3.set_xlabel('Time $t$', fontsize=12)
        ax3.set_ylabel('Absolute Error', fontsize=12, color='darkred')
        ax3.tick_params(axis='y', labelcolor='darkred')
        ax3.set_title(f'{optim_label} - Error Analysis (x=0.5)', fontsize=13)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_xlim([0, 1])
        ax3.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.3)
    plt.show()

# Main function
def train_1D_wave_CGMPINN_with_activations():
    # 1. Hyperparameter settings
    n_layers = 4              
    input_dim = 2             
    output_dim = 1          
    hidden_dim = 50        
    n_pde = 2000             
    n_initial = 300          
    n_boundary = 300         
    n_test = 100              
    lr_lbfgs_config = {'Tanh': 0.1}
    lr_adam_lbfgs = 0.8       
    epochs_adam = 15000    
    max_iter_lbfgs = 15000    
    adam_lbfgs_adam_epochs = 5000  
    adam_lbfgs_lbfgs_iter = 10000  
    
    # Curriculum learning + GMM parameters
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
    
    # Activation function dictionary
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (CGMPINN)', 'L-BFGS (CGMPINN)', 'Adam→L-BFGS (CGMPINN)']
    
    # 2. Generate data
    print("Generating training and testing data...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde,
        n_initial=n_initial,
        n_boundary=n_boundary,
        n_test=n_test
    )
    x_test, t_test = test_data
    
    # Calculate exact solution: u(x,t) = e^(-γt) * sin(α1πx) * cos(α2πt)
    u_exact = torch.exp(-GAMMA * t_test) * torch.sin(ALPHA1 * np.pi * x_test) * torch.cos(ALPHA2 * np.pi * t_test)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name} | Equation parameters: α1={ALPHA1}, α2={ALPHA2}, γ={GAMMA}, β={BETA}, c={c}")
        print("="*80)
        
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Adam training (CGMPINN)
        print(f"\n--- {activation_name} + Adam (CGMPINN+ReLoBRaLo) ---")
        model_adam = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs
        )
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        start_time = time.time()
        train_with_optimizer(
            model=model_adam,
            optimizer=optimizer_adam,
            epochs=epochs_adam,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam (CGMPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_initial_loss': final_initial,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 L-BFGS training (CGMPINN)
        print(f"\n--- {activation_name} + L-BFGS (CGMPINN) ---")
        model_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=False, relobralo_kwargs=relobralo_kwargs  # L-BFGS暂不启用ReLoBRaLo
        )
        loss_history_lbfgs = []
        start_time = time.time()
        train_with_lbfgs(
            model=model_lbfgs,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_lbfgs,
            lr=lr_lbfgs_config[activation_name],
            max_iter=max_iter_lbfgs
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate L-BFGS
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_initial_loss': final_initial,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Adam→L-BFGS training (CGMPINN)
        print(f"\n--- {activation_name} + Adam→L-BFGS (CGMPINN+ReLoBRaLo) ---")
        model_adam_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs
        )
        loss_history_adam_lbfgs = []
        start_time = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs,
            lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam→L-BFGS
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam→L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_initial_loss': final_initial,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        
        # Save Tanh+Adam→LBFGS model
        if activation_name == 'Tanh':  
            save_path = "1D_wave_tanh_adam2lbfgs_cgmpinn.pth"
            complete_save_dict = {
                'model_state_dict': model_adam_lbfgs.state_dict(), 
                'loss_history': loss_history_adam_lbfgs,            
                'final_total_loss': final_total,                    
                'training_time': train_time,                        
                'activation_name': activation_name,                
                'hyper_parameters': {                               
                    'n_layers': n_layers,
                    'hidden_dim': hidden_dim,
                    'adam_epochs': adam_lbfgs_adam_epochs,
                    'lbfgs_max_iter': adam_lbfgs_lbfgs_iter
                }
            }
            torch.save(complete_save_dict, save_path)
            print(f"✅ Tanh+Adam→L-BFGS wave equation CGMPINN model saved to: {save_path}")
        
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save current activation function results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # 5. Visualization of results
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    plot_damping_effect(
        x_test.numpy(), t_test.numpy(),
        {optim_label: activation_results['Tanh'][optim_label]['u_pred'] for optim_label in optim_labels},
        u_exact.numpy()
    )
    
    # 6. Print comprehensive comparison table
    print("\n" + "="*120)
    print("Comprehensive comparison table for all activation functions × optimizers (1D wave equation with damping - CGMPINN)")
    print("="*120)
    header = (f"{'Activation':<10} {'Optimizer':<20} {'Training Time(s)':<12} {'L2 Error':<12} "
              f"{'L2 Relative Error':<12} {'L∞ Error':<12} {'Final Total Loss':<12}")
    print(header)
    print("-"*120)
    for activation_name in activation_names:
        for idx, optim_label in enumerate(optim_labels):
            results = activation_results[activation_name][optim_label]
            train_time = activation_training_times[activation_name][idx]
            print(
                f"{activation_name:<10} "
                f"{optim_label:<20} "
                f"{train_time:<12.2f} "
                f"{results['l2_error']:<12.2e} "
                f"{results['l2_relative_error']:<12.2e} "
                f"{results['linf_error']:<12.2e} "
                f"{results['final_total_loss']:<12.2e}"
            )
    
    # 7. Print best performance for each activation function
    print("\n" + "="*100)
    print("Best performance for each activation function (sorted by L2 error - CGMPINN)")
    print("="*100)
    for activation_name in activation_names:
        print(f"\n【{activation_name} - CGMPINN】")
        optim_results = []
        for optim_label, results in activation_results[activation_name].items():
            train_time = activation_training_times[activation_name][optim_labels.index(optim_label)]
            optim_results.append({
                'optimizer': optim_label,
                'train_time': train_time,
                'l2_error': results['l2_error'],
                'linf_error': results['linf_error'],
                'final_loss': results['final_total_loss']
            })
        # Sort by L2 error in ascending order
        optim_results_sorted = sorted(optim_results, key=lambda x: x['l2_error'])
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<18} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_wave_CGMPINN_with_activations()