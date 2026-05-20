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

# Equation parameters
D = 0.25               # Diffusion coefficient
r = 4.0                # Reaction rate
LAMBDA = np.sqrt(r / (6 * D))  # Wavefront steepness parameter
c = 5 * np.sqrt(D * r / 6)     # Wave speed
x_min, x_max = -5, 5  # Spatial domain
t_min, t_max = 0, 2   # Temporal domain

class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

def fisher_kpp_exact_solution(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    z = LAMBDA * (x - c * t)
    # Numerical stability handling (avoid exp overflow)
    z_clamped = torch.clamp(z, min=-50, max=50)
    exp_z = torch.exp(z_clamped)
    u = 1.0 / (1.0 + exp_z) ** 2  
    return u

class AdaptiveLossWeights:
    """Adaptive loss weights base class (foundation for ReLoBRaLo)"""
    def __init__(self, n_losses=3, device='cpu'):
        super().__init__()
        self.n_losses = n_losses  
        self.device = device      
        self.weights = torch.ones(n_losses, device=device, dtype=torch.float32)  

class ReLoBRaLoWeights(AdaptiveLossWeights):
    def __init__(self, n_losses=3, alpha=0.999, temperature=1.0, 
                 rho=0.99, device='cpu'):
        super().__init__(n_losses, device)
        self.alpha = alpha           # Exponential moving average coefficient
        self.temperature = temperature  # softmax temperature
        self.rho = rho               # Random lookback probability
        self.loss_history = [[] for _ in range(n_losses)]  # Store historical losses
        self.ema_losses = None  # Exponential moving average losses
        self.initial_losses = None  # Initial losses (for normalization)
        
    def update(self, losses, **kwargs):
        losses_tensor = torch.tensor(losses, device=self.device, dtype=torch.float32)
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss)
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.clone()
            self.ema_losses = losses_tensor.clone()
            return
        self.ema_losses = self.alpha * self.ema_losses + (1 - self.alpha) * losses_tensor
        if np.random.rand() < self.rho:
            reference = self.ema_losses
        else:
            lookback_idx = np.random.randint(0, max(1, len(self.loss_history[0]) - 1))
            reference = torch.tensor(
                [self.loss_history[i][lookback_idx] for i in range(self.n_losses)],
                device=self.device, dtype=torch.float32
            )
        relative_losses = losses_tensor / (reference + 1e-8)
        scaled_losses = relative_losses / self.temperature
        self.weights = self.n_losses * torch.softmax(scaled_losses, dim=0)

class GMMCurriculumWeight:
    def __init__(self, n_components=4, update_interval=200, epsilon=1e-6, 
                 beta=1.0, tau_saturation=0.8, use_variance_factor=False):
        self.n_components = n_components
        self.update_interval = update_interval
        self.epsilon = epsilon
        self.beta = beta
        self.tau_saturation = tau_saturation
        self.use_variance_factor = use_variance_factor
        self.gmm = GaussianMixture(n_components=n_components, random_state=1224)
        self._last_valid_weights = None  

    def compute_weights(self, residuals: torch.Tensor, tau: float) -> torch.Tensor:
        res_np = residuals.detach().cpu().numpy().reshape(-1, 1)
        try:
            self.gmm.fit(res_np)
            gamma = self.gmm.predict_proba(res_np)  # Posterior probabilities (n_pde, K)
            sigma_sq = self.gmm.covariances_.flatten()  # Variances of Gaussian components
            # Compute component difficulty
            res_squared = res_np.flatten() ** 2
            component_difficulty = np.array([
                np.sum(gamma[:, j] * res_squared) / (np.sum(gamma[:, j]) + self.epsilon)
                for j in range(self.n_components)
            ])
            # Normalize difficulty to [0,1]
            if component_difficulty.max() - component_difficulty.min() > self.epsilon:
                normalized_diff = (component_difficulty - component_difficulty.min()) / (component_difficulty.max() - component_difficulty.min())
            else:
                normalized_diff = np.zeros_like(component_difficulty)
            # Curriculum learning weights: tau from 0→1, from easy to hard samples
            easy_weight = np.exp(-self.beta * normalized_diff)
            hard_weight = np.exp(-self.beta * (1 - normalized_diff))
            curriculum_weight = (1 - tau) * easy_weight + tau * hard_weight
            # Variance factor optimization (optional)
            if self.use_variance_factor:
                variance_factor = 1 / (sigma_sq + self.epsilon)
                variance_factor = variance_factor / variance_factor.max()
                effective_variance_factor = (1 - tau) * variance_factor + tau * 1.0
                component_weights = curriculum_weight * effective_variance_factor
            else:
                component_weights = curriculum_weight
            # Compute final weights for each collocation point
            point_weights = np.sum(gamma * component_weights[np.newaxis, :], axis=1)
            self._last_valid_weights = point_weights.copy()
        except Exception as e:
            # Use cached weights or all ones weights in case of exception
            if self._last_valid_weights is not None and len(self._last_valid_weights) == len(res_np):
                point_weights = self._last_valid_weights
            else:
                point_weights = np.ones(len(res_np))
        # Normalize weights to ensure mean is 1
        point_weights = point_weights / (point_weights.mean() + self.epsilon)
        return torch.tensor(point_weights.reshape(-1, 1), dtype=torch.float32).to(residuals.device)

# CGMPINN model
class CGMPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, 
                 use_curriculum=True, gmm_kwargs=None,
                 use_relobralo=True, relobralo_kwargs=None):
        super(CGMPINN, self).__init__()
        self.activation = activation
        self.use_curriculum = use_curriculum  # GMM curriculum learning switch
        self.use_relobralo = use_relobralo    # ReLoBRaLo adaptive weight switch
        self.total_train_steps = 0  # Total training steps
        self.current_tau = 0.0     # Training progress (0=early, 1=late)

        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

        def _init_weights(m):  
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)  

        # Initialize GMM curriculum learning
        if self.use_curriculum:
            gmm_kwargs = gmm_kwargs or {}
            self.curriculum_weight = GMMCurriculumWeight(**gmm_kwargs)
            self.sample_weights = None  # Cache for collocation point weights
        
        self.latest_pde_loss = None
        self.latest_initial_loss = None
        self.latest_boundary_loss = None

        # Initialize ReLoBRaLo adaptive loss weights
        if self.use_relobralo:
            relobralo_kwargs = relobralo_kwargs or {}
            model_device = next(self.parameters()).device
            relobralo_kwargs['device'] = model_device
            self.relobralo = ReLoBRaLoWeights(**relobralo_kwargs)

    def update_curriculum_weights(self, x: torch.Tensor, t: torch.Tensor) -> None:
        if not self.use_curriculum:
            return
        u, u_t, u_xx = self.compute_gradients(x, t)
        pde_residual = u_t - D * u_xx - r * u * (1 - u)
        self.sample_weights = self.curriculum_weight.compute_weights(pde_residual, self.current_tau)

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        if total_steps == 0:
            self.current_tau = 0.0
        else:
            tau_base = total_steps * self.curriculum_weight.tau_saturation
            self.current_tau = min(current_step / tau_base, 1.0)  # tau≤1

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_norm = x / 5.0
        t_norm = t - 1.0 
        
        input_tensor = torch.cat([x_norm, t_norm], dim=1)
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
        return u, u_t, u_xx
    
    def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u, u_t, u_xx = self.compute_gradients(x, t)
        pde_residual = u_t - D * u_xx - r * u * (1 - u)
        if self.use_curriculum and self.sample_weights is not None:
            weighted_residual = self.sample_weights.detach() * (pde_residual ** 2)
            pde_loss_val = torch.mean(weighted_residual)
        else:
            pde_loss_val = torch.mean(pde_residual ** 2)
        self.latest_pde_loss = pde_loss_val.item()
        return pde_loss_val

    def initial_loss(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u_exact = fisher_kpp_exact_solution(x, t0)  
        initial_residual = u_pred - u_exact
        initial_loss_val = torch.mean(initial_residual ** 2)
        self.latest_initial_loss = initial_loss_val.item()
        return initial_loss_val

    def boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        x_left = torch.full_like(t, x_min, requires_grad=True)
        u_left_pred = self.forward(x_left, t)
        u_left_exact = fisher_kpp_exact_solution(x_left, t)  
        
        x_right = torch.full_like(t, x_max, requires_grad=True)
        u_right_pred = self.forward(x_right, t)
        u_right_exact = fisher_kpp_exact_solution(x_right, t) 
        
        boundary_loss_val = torch.mean((u_left_pred - u_left_exact) ** 2) + \
                           torch.mean((u_right_pred - u_right_exact) ** 2)
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
            return weights[0] * pde_loss_val + weights[1] * initial_loss_val + weights[2] * boundary_loss_val
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
                total_loss_val = weights[0] * pde_loss_val + weights[1] * initial_loss_val + weights[2] * boundary_loss_val
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
    # PDE interior points: x ∈ [-5,5], t ∈ [0,2] randomly distributed
    x_pde = (x_max - x_min) * torch.rand(n_pde, 1) + x_min
    t_pde = (t_max - t_min) * torch.rand(n_pde, 1) + t_min
    x_pde.requires_grad_(True)
    t_pde.requires_grad_(True)
    pde_data = (x_pde, t_pde)
    
    # Initial condition points: t=0, x ∈ [-5,5]
    x_initial = (x_max - x_min) * torch.rand(n_initial, 1) + x_min
    t_initial = torch.zeros_like(x_initial)
    x_initial.requires_grad_(True)
    t_initial.requires_grad_(True)
    initial_data = (x_initial, t_initial)
    
    # Boundary condition points: t ∈ [0,2]
    t_boundary = (t_max - t_min) * torch.rand(n_boundary, 1) + t_min
    t_boundary.requires_grad_(True)
    boundary_data = t_boundary
    
    # Test data: uniform grid
    x_test = torch.linspace(x_min, x_max, n_test).reshape(-1, 1)
    t_test = torch.linspace(t_min, t_max, n_test).reshape(-1, 1)
    x_test_grid, t_test_grid = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    x_test_flat = x_test_grid.reshape(-1, 1)
    t_test_flat = t_test_grid.reshape(-1, 1)
    test_data = (x_test_flat, t_test_flat)
    
    return pde_data, initial_data, boundary_data, test_data

# Adam training function
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
            if epoch % 1000 == 0:
                print(f"  → Epoch {epoch+1}, tau={model.current_tau:.3f}, updated GMM sampling weights")

        # Compute weighted total loss and backpropagate
        total_loss = model.compute_weighted_total_loss(pde_data, initial_data, boundary_data)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        # Print training log
        if (epoch + 1) % 1000 == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            relobralo_info = ""
            pde_loss = model.latest_pde_loss or 0.0
            initial_loss = model.latest_initial_loss or 0.0
            boundary_loss = model.latest_boundary_loss or 0.0
            if model.use_relobralo:
                weights = model.relobralo.weights.detach().cpu().numpy()
                relobralo_info = f" | ReLoBRaLo weights (PDE/IC/BC): [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]"
            print(
                f'Epoch {epoch+1:4d}{curr_info}{relobralo_info} | Total loss: {total_loss.item():.2e} | '
                f'PDE loss: {pde_loss:.2e} | Initial loss: {initial_loss:.2e} | Boundary loss: {boundary_loss:.2e}'
            )
    
    return step_offset + epochs

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
        pde_loss_val = model.latest_pde_loss or 0.0
        initial_loss_val = model.latest_initial_loss or 0.0
        boundary_loss_val = model.latest_boundary_loss or 0.0
        return total_loss

    optimizer = optim.LBFGS(
        model.parameters(), max_iter=1, max_eval=10,
        line_search_fn='strong_wolfe', lr=lr
    )

    for iter_idx in range(max_iter):
        global_step = step_offset + iter_idx + 1
        model.set_training_progress(global_step, global_total_steps)
        
        if model.use_curriculum and (iter_idx % model.curriculum_weight.update_interval == 0):
            model.update_curriculum_weights(x_pde, t_pde)
            if iter_idx % 500 == 0:
                print(f"  → Iteration {iter_idx+1}, tau={model.current_tau:.3f}, updated GMM sampling weights")

        total_loss = optimizer.step(closure)
        loss_history.append(total_loss.item())

        # Print training log
        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            print(
                f'Iteration {iter_idx+1}/{max_iter}{curr_info} | Total loss: {total_loss.item():.2e} | '
                f'PDE loss: {pde_loss_val:.2e} | Initial loss: {initial_loss_val:.2e} | Boundary loss: {boundary_loss_val:.2e}'
            )
    
    return step_offset + max_iter

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
    print(f"\n=== Stage 1: Adam Global Exploration (ReLoBRaLo+GMM, tau: 0 → {adam_epochs/global_total_steps:.3f}) ===")
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    current_step = train_with_optimizer(
        model=model, optimizer=optimizer_adam, epochs=adam_epochs,
        pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
        loss_history=loss_history, step_offset=0, global_total_steps=global_total_steps
    )
    
    print(f"\n=== Stage 2: L-BFGS Local Refinement (GMM, tau: {current_step/global_total_steps:.3f} → 1.0) ===")
    train_with_lbfgs(
        model=model, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
        loss_history=loss_history, lr=lr_lbfgs, max_iter=lbfgs_max_iter,
        step_offset=current_step, global_total_steps=global_total_steps
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
        fig.suptitle(f'Fisher-KPP Solution Comparison - Activation: {activation_name} (CGMPINN)', fontsize=24, y=0.99)
        
        for idx, optim_label in enumerate(optim_labels):
            u_pred = results[optim_label]['u_pred']
            u_pred_grid = u_pred.reshape(n_test, n_test)
            u_exact_grid = u_exact_np.reshape(n_test, n_test)
            error_grid = np.abs(u_pred_grid - u_exact_grid)
            x_grid = x_test.reshape(n_test, n_test)
            t_grid = t_test.reshape(n_test, n_test)
            
            ax1 = axes[idx, 0]
            im1 = ax1.pcolormesh(t_grid, x_grid, u_pred_grid, cmap=cm.jet, shading='gouraud', vmin=0, vmax=1)
            ax1.set_xlabel('Time (t)', fontsize=10)
            ax1.set_ylabel('Position (x)', fontsize=10)
            ax1.set_title(f'{optim_label} - Predicted', fontsize=12)
            plt.colorbar(im1, ax=ax1, shrink=0.6, aspect=5)
            
            ax2 = axes[idx, 1]
            im2 = ax2.pcolormesh(t_grid, x_grid, u_exact_grid, cmap=cm.jet, shading='gouraud', vmin=0, vmax=1)
            ax2.set_title('Traveling Wave Exact Solution', fontsize=12)
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

# Main training function
def train_1D_fisher_kpp_CGMPINN():
    # Hyperparameter settings
    n_layers = 4               
    input_dim = 2              
    output_dim = 1             
    hidden_dim = 80
    n_pde = 8000               
    n_initial = 400            
    n_boundary = 400           
    n_test = 100               
    lr_lbfgs_config = {'Tanh': 1.0}  
    lr_adam_lbfgs = 0.8        
    epochs_adam = 20000         
    max_iter_lbfgs = 20000      
    adam_lbfgs_adam_epochs = 8000   
    adam_lbfgs_lbfgs_iter = 12000   

    # GMM curriculum learning parameters
    gmm_kwargs = {
        'n_components': 4,
        'update_interval': 100,
        'epsilon': 1e-6,
        'beta': 1.0,
        'tau_saturation': 0.8,
        'use_variance_factor': True
    }
    
    # ReLoBRaLo adaptive weighting parameters
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
    
    # Generate training/testing data for Fisher-KPP
    print("Generating training/testing data for Fisher-KPP equation...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde, n_initial=n_initial, n_boundary=n_boundary, n_test=n_test
    )
    x_test, t_test = test_data
    
    # Compute traveling wave exact solution for Fisher-KPP
    u_exact = fisher_kpp_exact_solution(x_test, t_test)
    u_exact_np = u_exact.numpy()
    
    # Initialize result storage
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation Function: {activation_name} | Fisher-KPP Parameters: D={D}, r={r}, c={c:.2f}")
        print("="*80)
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 7.1 Pure Adam training (CGMPINN+ReLoBRaLo+GMM)
        print(f"\n--- {activation_name} + Adam (CGMPINN) ---")
        model_adam = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers, activation_fn,
            use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs
        )
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        start_time = time.time()
        train_with_optimizer(
            model=model_adam, optimizer=optimizer_adam, epochs=epochs_adam,
            pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        # Evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 7.2 Pure L-BFGS training (CGMPINN+GMM, ReLoBRaLo off)
        print(f"\n--- {activation_name} + L-BFGS (CGMPINN) ---")
        model_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers, activation_fn,
            use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=False, relobralo_kwargs=relobralo_kwargs
        )
        loss_history_lbfgs = []
        start_time = time.time()
        train_with_lbfgs(
            model=model_lbfgs, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_lbfgs, lr=lr_lbfgs_config[activation_name], max_iter=max_iter_lbfgs
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        # Evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 7.3 Adam→L-BFGS two-stage training (CGMPINN+ReLoBRaLo+GMM)
        print(f"\n--- {activation_name} + Adam→L-BFGS (CGMPINN) ---")
        model_adam_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers, activation_fn,
            use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs # ReLoBRaLo optionally enabled
        )
        loss_history_adam_lbfgs = []
        start_time = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs, adam_epochs=adam_lbfgs_adam_epochs,
            lr_lbfgs=lr_adam_lbfgs, lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        # Evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam→L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        
        # Save the best model (Tanh+Adam→L-BFGS)
        save_path = "1D_fisher_kpp_tanh_adam2lbfgs_cgmpinn.pth"
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
        print(f"✅ Fisher-KPP CGMPINN best model saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save all results for the current activation function
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # Visualize results
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # Print comprehensive comparison table
    print("\n" + "="*120)
    print("Fisher-KPP equation CGMPINN training comprehensive comparison table (Activation Function × Optimizer)")
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
    
    # Print best performances
    print("\n" + "="*100)
    print("Best performances of each optimizer (sorted by standard L2 error)")
    print("="*100)
    for activation_name in activation_names:
        print(f"\n[{activation_name}]")
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
                f"  Rank {i}: {res['optimizer']:<20} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_fisher_kpp_CGMPINN()