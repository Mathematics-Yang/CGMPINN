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

# set random seed for reproducibility
torch.manual_seed(1224)
np.random.seed(1224)

# Define equation parameters
ALPHA1 = 5.0  # Sine term parameter
ALPHA2 = 3.0  # Cosine term parameter
k = 20.0      # Steepness of hyperbolic tangent function

class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# Adaptive loss weights base class
class AdaptiveLossWeights:
    """
    Base class for adaptive loss weights (provides foundation for ReLoBRaLo)
    """
    def __init__(self, n_losses=2, device='cpu'):
        super().__init__()
        self.n_losses = n_losses  # Number of loss components 
        self.device = device      # Computing device
        self.weights = torch.ones(n_losses, device=device, dtype=torch.float32)  # Initialize weights to 1

class ReLoBRaLoWeights(AdaptiveLossWeights):
    """
    ReLoBRaLo: Relative Loss Balancing with Random Lookback
    Adaptive weights based on loss change rates
    Core ideas:
    - Loss components that decrease slowly should receive higher weights
    - Use random lookback to smooth weight updates
    """
    def __init__(self, n_losses=2, alpha=0.999, temperature=1.0, 
                 rho=0.99, device='cpu'):
        super().__init__(n_losses, device)
        self.alpha = alpha           # Exponential moving average coefficient
        self.temperature = temperature  # Softmax temperature
        self.rho = rho               # Random lookback probability
        
        # Store historical losses
        self.loss_history = [[] for _ in range(n_losses)]
        self.ema_losses = None  # Exponential moving average of losses
        self.initial_losses = None  # Initial losses (for normalization)
        
    def update(self, losses, **kwargs):
        """
        Update weights
        :param losses: Current list of loss values [pde_loss, boundary_loss] 
        """
        losses_tensor = torch.tensor(losses, device=self.device, dtype=torch.float32)
        
        # record history
        for i, loss in enumerate(losses):
            self.loss_history[i].append(loss)
        
        # Initialization
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.clone()
            self.ema_losses = losses_tensor.clone()
            return
        
        # Update EMA losses
        self.ema_losses = self.alpha * self.ema_losses + (1 - self.alpha) * losses_tensor
        
        # Calculate relative loss change rates
        # Random lookback: use EMA with probability rho, otherwise use current value
        if np.random.rand() < self.rho:
            reference = self.ema_losses
        else:
            # Randomly select a point from history
            lookback_idx = np.random.randint(0, max(1, len(self.loss_history[0]) - 1))
            reference = torch.tensor(
                [self.loss_history[i][lookback_idx] for i in range(self.n_losses)],
                device=self.device, dtype=torch.float32
            )
        
        # Calculate relative change rates
        relative_losses = losses_tensor / (reference + 1e-8)
        
        # Use softmax to calculate weights
        scaled_losses = relative_losses / self.temperature
        self.weights = self.n_losses * torch.softmax(scaled_losses, dim=0)

# GMM weight calculation utility class (implements curriculum learning from easy to hard)
class GMMCurriculumWeight:
    def __init__(self, n_components=4, update_interval=200, epsilon=1e-6, 
                 beta=1.0,  # Increase beta to enhance distinction
                 tau_saturation=0.8,
                 use_variance_factor=False):  # Whether to use variance factor
        """
        Initialize GMM curriculum learning weight calculator (from easy to hard, improved version)
        :param n_components: Number of Gaussian components K
        :param update_interval: Weight update interval
        :param epsilon: Small value to prevent division by zero
        :param beta: Curriculum learning intensity coefficient (larger beta means more drastic difficulty switching)
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
        self._last_valid_weights = None  # Initialize valid weights cache

    def compute_weights(self, residuals: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Calculate weights based on residuals and training progress (core: tau controls the transition from easy to hard, improved version)
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
            
            # 2. Use posterior probabilities to weight component difficulty (more accurate difficulty assessment)
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
            easy_weight = np.exp(-self.beta * normalized_diff)  # Easy sample weights: lower difficulty → higher weight
            hard_weight = np.exp(-self.beta * (1 - normalized_diff))  # Hard sample weights: higher difficulty → higher weight
            
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
        
        # 8. Normalize weights (ensure mean is 1, do not change overall loss scale)
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
        self.total_train_steps = 0  # Record total training steps (for calculating tau)
        self.current_tau = 0.0     # Current training progress

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
        self.latest_boundary_loss = None

        # Initialize ReLoBRaLo adaptive loss weight calculator
        if self.use_relobralo:
            relobralo_kwargs = relobralo_kwargs or {}
            model_device = next(self.parameters()).device
            relobralo_kwargs['device'] = model_device 
            relobralo_kwargs['n_losses'] = 2  
            self.relobralo = ReLoBRaLoWeights(**relobralo_kwargs)

    def update_curriculum_weights(self, x: torch.Tensor) -> None:
        if not self.use_curriculum:
            return
        u_xx = self.compute_gradients(x)  
        pde_residual = u_xx - self.source_term(x)  
        # Calculate weights based on current training progress tau
        self.sample_weights = self.curriculum_weight.compute_weights(pde_residual, self.current_tau)

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        if total_steps == 0:
            self.current_tau = 0.0
        else:
            # Adapt GMM tau saturation point (default: tau reaches 1 at 80% of total steps)
            tau_base = total_steps * self.curriculum_weight.tau_saturation
            self.current_tau = min(current_step / tau_base, 1.0)  # Limit tau ≤ 1

    # Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    # Compute gradients
    def compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        u = self.forward(x)
        u_x = torch.autograd.grad(
            outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        return u_xx
    
    # Analytical solution
    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(ALPHA1 * np.pi * x) * torch.cos(ALPHA2 * np.pi * x) + torch.tanh(k * x)

    # Define source term f(x)
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        pi = np.pi
        alpha1pi = ALPHA1 * pi
        alpha2pi = ALPHA2 * pi
        term1_second = - (alpha1pi**2 + alpha2pi**2) * torch.sin(alpha1pi * x) * torch.cos(alpha2pi * x) \
                        - 2 * alpha1pi * alpha2pi * torch.cos(alpha1pi * x) * torch.sin(alpha2pi * x)
        sech_kx = 1 / torch.cosh(k * x)
        term2_second = -2 * k**2 * sech_kx**2 * torch.tanh(k * x)
        f = term1_second + term2_second
        return f

    def pde_loss(self, x: torch.Tensor) -> torch.Tensor:
        u_xx = self.compute_gradients(x)
        f = self.source_term(x)
        pde_residual = u_xx - f  

        if self.use_curriculum and self.sample_weights is not None:
            # Weighted loss: simple samples have higher weights initially, gradually decreasing later 
            weighted_residual = self.sample_weights.detach() * (pde_residual ** 2)
            pde_loss_val = torch.mean(weighted_residual)
        else:
            pde_loss_val = torch.mean(pde_residual ** 2)
        
        # Cache latest PDE loss value
        self.latest_pde_loss = pde_loss_val.item()
        
        return pde_loss_val
    
    def boundary_loss(self) -> torch.Tensor:
        x0 = torch.tensor([[0.0]], requires_grad=True).to(next(self.parameters()).device)
        u0_pred = self.forward(x0)
        u0_exact = self.analytical_solution(x0)
        x1 = torch.tensor([[1.0]], requires_grad=True).to(next(self.parameters()).device)
        u1_pred = self.forward(x1)
        u1_exact = self.analytical_solution(x1)
        boundary_residual1 = u0_pred - u0_exact
        boundary_residual2 = u1_pred - u1_exact
        boundary_loss_val = torch.mean(boundary_residual1 ** 2) + torch.mean(boundary_residual2 ** 2)
        self.latest_boundary_loss = boundary_loss_val.item()
        
        return boundary_loss_val
    
    def compute_weighted_total_loss(self, pde_data) -> torch.Tensor:
        x_pde = pde_data
        pde_loss_val = self.pde_loss(x_pde)
        boundary_loss_val = self.boundary_loss()
        if self.use_relobralo:
            current_losses = [self.latest_pde_loss, self.latest_boundary_loss]
            self.relobralo.update(current_losses)
            weights = self.relobralo.weights.detach()
            return weights[0] * pde_loss_val + weights[1] * boundary_loss_val
        else:
            return pde_loss_val + boundary_loss_val
    
    # Compute final total loss
    def final_total_loss(self, pde_data) -> tuple[float, float, float]:
        x_pde = pde_data
        
        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde).item()
            boundary_loss_val = self.boundary_loss().item()
            
            if self.use_relobralo:
                weights = self.relobralo.weights.detach().cpu().numpy()
                total_loss_val = weights[0] * pde_loss_val + weights[1] * boundary_loss_val
            else:
                total_loss_val = pde_loss_val + boundary_loss_val
        
        return total_loss_val, pde_loss_val, boundary_loss_val

# Generate training and testing data
def generate_data(
    n_pde: int,
    n_test: int 
) -> tuple:
    x_pde = torch.rand(n_pde, 1, requires_grad=True)
    pde_data = x_pde
    x_test = torch.linspace(0.0, 1.0, n_test).reshape(-1, 1)
    test_data = x_test
    
    return pde_data, test_data

# Training function (Adam)
def train_with_optimizer(
    model: CGMPINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: torch.Tensor,
    loss_history: list,
    step_offset: int = 0,           # Step offset
    global_total_steps: int = None  # Global total steps
) -> int:  # Return cumulative steps
    x_pde = pde_data
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
            model.update_curriculum_weights(x_pde)
            print(f"  → Epoch {epoch+1} (Global step {global_step}), current curriculum weights (tau={model.current_tau:.3f})")

        # Compute weighted total loss using ReLoBRaLo
        total_loss = model.compute_weighted_total_loss(pde_data)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if (epoch + 1) % 1000 == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            relobralo_info = ""

            # Retrieve cached component losses
            pde_loss = model.latest_pde_loss if model.latest_pde_loss is not None else 0.0
            boundary_loss = model.latest_boundary_loss if model.latest_boundary_loss is not None else 0.0

            if model.use_relobralo:
                weights = model.relobralo.weights.detach().cpu().numpy()
                relobralo_info = f" | ReLoBRaLo weights (PDE/BC): [{weights[0]:.2e}, {weights[1]:.2e}]"
            print(
                f'Epoch {epoch+1:4d}{curr_info}{relobralo_info} | Total loss: {total_loss.item():.2e} | '
                f'PDE loss: {pde_loss:.2e} | Boundary condition loss: {boundary_loss:.2e}'
            )
    
    # Return cumulative global steps
    return step_offset + epochs

# L-BFGS training function
def train_with_lbfgs(
    model: CGMPINN,
    pde_data: torch.Tensor,
    loss_history: list,
    lr: float,
    max_iter: int,
    step_offset: int = 0,           # Step offset
    global_total_steps: int = None  # Global total steps
) -> int:  # Return cumulative steps
    x_pde = pde_data
    model.train()

    # If global total steps not specified, use current phase's max_iter
    if global_total_steps is None:
        global_total_steps = max_iter

    pde_loss_val = boundary_loss_val = 0.0

    # Initialize curriculum learning weights
    if model.use_curriculum:
        model.update_curriculum_weights(x_pde)

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, boundary_loss_val
        optimizer.zero_grad()
        # Compute weighted total loss using ReLoBRaLo
        total_loss = model.compute_weighted_total_loss(pde_data)
        total_loss.backward()
        
        # Cache component loss values (for printing)
        pde_loss_val = model.latest_pde_loss if model.latest_pde_loss is not None else 0.0
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
            model.update_curriculum_weights(x_pde)
            print(f"  → Iteration {iter_idx+1} (Global step {global_step}), current curriculum weight (tau={model.current_tau:.3f})")

        loss_history.append(total_loss.item())

        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            relobralo_info = ""
            if model.use_relobralo:
                weights = model.relobralo.weights.detach().cpu().numpy()
                relobralo_info = f" | ReLoBRaLo weights (PDE/BC): [{weights[0]:.2e}, {weights[1]:.2e}]"
            print(
                f'Iteration {iter_idx+1}/{max_iter}{curr_info}{relobralo_info} | Total loss: {total_loss.item():.2e} | '
                f'PDE loss: {pde_loss_val:.2e} | Boundary condition loss: {boundary_loss_val:.2e}'
            )
    
    # Return cumulative global steps
    return step_offset + max_iter

# Two-phase training function (Adam→L-BFGS)
def train_adam_lbfgs(
    model: CGMPINN,
    pde_data: torch.Tensor,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    # Calculate global total steps (Adam + L-BFGS)
    global_total_steps = adam_epochs + lbfgs_max_iter
    
    print(f"\n=== Phase 1: Adam Global Exploration (Starting from simple samples) ===")
    print(f"    Global step range: 1 ~ {adam_epochs}, tau: 0 → {adam_epochs/global_total_steps:.3f}")
    
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    current_step = train_with_optimizer(
        model=model, 
        optimizer=optimizer_adam, 
        epochs=adam_epochs,
        pde_data=pde_data,
        loss_history=loss_history,
        step_offset=0,                        # Adam starts from 0
        global_total_steps=global_total_steps # Use global total steps
    )
    
    print(f"\n=== Phase 2: L-BFGS Local Fine-tuning (Gradually focusing on difficult samples) ===")
    print(f"    Global step range: {current_step+1} ~ {global_total_steps}, tau: {current_step/global_total_steps:.3f} → 1.0")
    
    train_with_lbfgs(
        model=model, 
        pde_data=pde_data,
        loss_history=loss_history, 
        lr=lr_lbfgs, 
        max_iter=lbfgs_max_iter,
        step_offset=current_step,             # L-BFGS continues from where Adam ended
        global_total_steps=global_total_steps # Use global total steps
    )

# Evaluation function
def evaluate_model(
    model: CGMPINN,
    test_data: torch.Tensor,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    x_test = test_data
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test)
        
        # L2 error
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        
        # L2 relative error
        u_exact_l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()
        l2_relative_error = l2_error / u_exact_l2_norm if u_exact_l2_norm > 1e-10 else float('inf')
        
        # L∞ error
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        
        # Pointwise error and predictions
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
    test_data: torch.Tensor,
    u_exact_np: np.ndarray,
    optim_labels: list
) -> None:
    x_test_np = test_data.numpy()
    
    for activation_name, results in activation_results.items():
        fig, axes = plt.subplots(len(optim_labels), 2, figsize=(14, 4*len(optim_labels)))
        fig.suptitle(f'Solution Comparison - Activation: {activation_name}', fontsize=18, y=0.98)
        
        for idx, optim_label in enumerate(optim_labels):
            u_pred = results[optim_label]['u_pred']
            pointwise_error = results[optim_label]['pointwise_error']
            
            # Solution comparison
            ax1 = axes[idx, 0] if len(optim_labels) > 1 else axes[0]
            ax1.plot(x_test_np, u_exact_np, 'b-', label='Analytical', linewidth=2)
            ax1.plot(x_test_np, u_pred, 'r--', label='CGMPINN Prediction', linewidth=2, alpha=0.8)
            ax1.set_xlabel('x', fontsize=10)
            ax1.set_ylabel('u(x)', fontsize=10)
            ax1.set_title(f'{optim_label} - Solution', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Pointwise error
            ax2 = axes[idx, 1] if len(optim_labels) > 1 else axes[1]
            ax2.plot(x_test_np, pointwise_error, 'g-', linewidth=2)
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('Absolute Error', fontsize=10)
            ax2.set_title(f'{optim_label} - Pointwise Error', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

# Main training function
def train_1D_poisson_CGMPINN_with_activations_and_source_term():
    # Hyperparameter settings
    n_layers = 4
    input_dim = 1  
    output_dim = 1
    hidden_dim = 50
    n_pde = 1500
    n_test = 200
    lr_lbfgs_config = {'Tanh': 1.0}
    lr_adam_lbfgs_config = {'Tanh': 0.8}
    epochs_sgd_adam = 15000 
    max_iter_lbfgs = 15000   
    adam_lbfgs_adam_epochs = 5000
    adam_lbfgs_lbfgs_iter = 10000

    # Curriculum learning + GMM parameters
    gmm_kwargs = {
        'n_components': 4, # Use 4 Gaussian components
        'update_interval': 200,  # Update weights every 200 steps
        'epsilon': 1e-6,
        'beta': 1.0,  # Increase beta to enhance difficulty differentiation
        'tau_saturation': 0.8,  # Tau saturates at 80% of steps
        'use_variance_factor': True  # Enable variance factor to improve early training stability
    }
    
    # ReLoBRaLo adaptive weight parameters
    relobralo_kwargs = {
        'n_losses': 2,        # 2 loss components: PDE/boundary
        'alpha': 0.999,       # Exponential moving average coefficient, representing the weight of historical information, usually close to 1
        'temperature': 1.0,   # Softmax temperature (smaller means more pronounced weight differences)
        'rho': 0.99,          # Random rollback probability, usually close to 1
        'device': 'cpu'       # Device (change to 'cuda' if GPU is available)
    }

    # activation functions and labels
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (CGMPINN)', 'L-BFGS (CGMPINN)', 'Adam→L-BFGS (CGMPINN)']

    # Generate data
    print("Generating training and testing data...")
    pde_data, test_data = generate_data(n_pde=n_pde, n_test=n_test)
    
    # Calculate analytical solution
    model_temp = CGMPINN(input_dim, output_dim, hidden_dim, 1, TanhActivation())
    u_exact = model_temp.analytical_solution(test_data)
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
        print(f"\n--- {activation_name} + Adam (CGMPINN) ---")
        model_adam = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs    # ReLoBRaLo optionally enabled
        )
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        start_time_adam = time.time()
        train_with_optimizer(
            model=model_adam, optimizer=optimizer_adam, epochs=epochs_sgd_adam,
            pde_data=pde_data, loss_history=loss_history_adam
        )
        training_time_adam = time.time() - start_time_adam
        current_training_times.append(training_time_adam)

        # Evaluate Adam
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam.final_total_loss(pde_data)
        current_results['Adam (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary, 'u_pred': u_pred,
            'pointwise_error': pointwise_err
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam training time: {training_time_adam:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")

        # 2. L-BFGS (GMM + Curriculum Learning)
        print(f"\n--- {activation_name} + L-BFGS (CGMPINN) ---")
        model_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=False, relobralo_kwargs=relobralo_kwargs  # L-BFGS does not enable ReLoBRaLo for now
        )
        loss_history_lbfgs = []
        start_time_lbfgs = time.time()
        train_with_lbfgs(
            model=model_lbfgs, pde_data=pde_data,
            loss_history=loss_history_lbfgs, lr=lr_lbfgs_config[activation_name], max_iter=max_iter_lbfgs
        )
        training_time_lbfgs = time.time() - start_time_lbfgs
        current_training_times.append(training_time_lbfgs)

        # Evaluate L-BFGS
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_lbfgs.final_total_loss(pde_data)
        current_results['L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary, 'u_pred': u_pred,
            'pointwise_error': pointwise_err
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS training time: {training_time_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")

        # 3. Adam→L-BFGS (GMM + Curriculum Learning + ReLoBRaLo)
        print(f"\n--- {activation_name} + Adam→L-BFGS (CGMPINN) ---")
        model_adam_lbfgs = CGMPINN(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs  # ReLoBRaLo optionally enabled
        )
        loss_history_adam_lbfgs = []
        start_time_adam_lbfgs = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs, pde_data=pde_data,
            loss_history=loss_history_adam_lbfgs, lr_lbfgs=lr_adam_lbfgs_config[activation_name],
            adam_epochs=adam_lbfgs_adam_epochs, lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        training_time_adam_lbfgs = time.time() - start_time_adam_lbfgs
        current_training_times.append(training_time_adam_lbfgs)

        # Evaluate Adam→L-BFGS
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam_lbfgs.final_total_loss(pde_data)
        current_results['Adam→L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary, 'u_pred': u_pred,
            'pointwise_error': pointwise_err
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        if activation_name == 'Tanh':  
            # Define save path
            save_path = "1D_poisson_tanh_adam2lbfgs_cgmpinn.pth"
    
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
            print(f"✅ Tanh+Adam→L-BFGS complete information saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {training_time_adam_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")

        # Save results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times

    # Visualization
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_optimizer(activation_results, test_data, u_exact_np, optim_labels)

    # Print comprehensive comparison table
    print("\n" + "="*140)
    print("All activations × GMM + Curriculum Learning + ReLoBRaLo adaptive weights × Optimizers Comprehensive Comparison Table (Poisson Equation)")
    print("="*140)
    header = (f"{'Activation':<10} {'Optimizer':<25} {'Training Time(s)':<12} "
              f"{'Standard L2 Error':<12} {'L2 Relative Error':<12} {'L∞ Error':<12} {'Final Total Loss':<12}")
    print(header)
    print("-"*140)
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

    # Print best performances
    print("\n" + "="*110)
    print("Best performances of each activation (GMM + Curriculum Learning + ReLoBRaLo) sorted by Standard L2 Error (Poisson Equation)")
    print("="*110)
    for activation_name in activation_names:
        print(f"\n【{activation_name} - GMM + Curriculum Learning + ReLoBRaLo adaptive weights】")
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
                f"  No.{i}: {res['optimizer']:<25} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_poisson_CGMPINN_with_activations_and_source_term()