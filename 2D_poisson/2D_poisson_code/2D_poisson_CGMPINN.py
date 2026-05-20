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

class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# define constants
BETA1 = 3.0  
BETA2 = 2.0  
DOMAIN_MIN = 0.0  
DOMAIN_MAX = 1.0  

# adaptive loss weights base class
class AdaptiveLossWeights:
    """
    Base class for adaptive loss weights (foundation for ReLoBRaLo)
    """
    def __init__(self, n_losses=2, device='cpu'):
        super().__init__()
        self.n_losses = n_losses  # number of loss components 
        self.device = device      # computation device
        self.weights = torch.ones(n_losses, device=device, dtype=torch.float32)  # initialize weights to 1

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
        :param losses: current loss values list [pde_loss, boundary_loss]
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

# GMM weight computation utility class (implements curriculum learning from easy to hard)
class GMMCurriculumWeight:
    def __init__(self, n_components=4, update_interval=200, epsilon=1e-6, 
                 beta=1.0, tau_saturation=0.8, use_variance_factor=False):
        """
        Initialize GMM curriculum weight calculator (from easy to hard)
        :param n_components: number of Gaussian components K
        :param update_interval: weight update interval
        :param epsilon: small value to prevent division by zero
        :param beta: strength coefficient for curriculum learning (larger beta means more abrupt difficulty switching)
        :param tau_saturation: tau saturation point (default 0.8, meaning tau reaches 1 at 80% of the steps)
        :param use_variance_factor: whether to use variance factor in weight calculation
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
            easy_weight = np.exp(-self.beta * normalized_diff)  # Easy sample weights: lower difficulty → higher weight
            hard_weight = np.exp(-self.beta * (1 - normalized_diff))  # Hard sample weights: higher difficulty → higher weight
            
            # 5. Smooth transition controlled by tau (from easy to hard)
            curriculum_weight = (1 - tau) * easy_weight + tau * hard_weight
            
            # 6. Optional variance factor
            if self.use_variance_factor:
                variance_factor = 1 / (sigma_sq + self.epsilon)
                variance_factor = variance_factor / variance_factor.max()  # Normalize variance factor
                effective_variance_factor = (1 - tau) * variance_factor + tau * 1.0
                component_weights = curriculum_weight * effective_variance_factor
            else:
                component_weights = curriculum_weight
            
            # 7. Sample weights = weighted sum of posterior probabilities × component weights
            point_weights = np.sum(gamma * component_weights[np.newaxis, :], axis=1)
            self._last_valid_weights = point_weights.copy()  # Update last valid weights cache
        
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
class CGMPINN2D(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, 
                 use_curriculum=True, gmm_kwargs=None,
                 use_relobralo=True, relobralo_kwargs=None):
        super(CGMPINN2D, self).__init__()
        self.activation = activation
        self.use_curriculum = use_curriculum  # Curriculum learning switch
        self.use_relobralo = use_relobralo    # ReLoBRaLo adaptive weight switch
        self.total_train_steps = 0  # Record total training steps
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
        self.latest_boundary_loss = None

        # Initialize ReLoBRaLo adaptive loss weight calculator
        if self.use_relobralo:
            relobralo_kwargs = relobralo_kwargs or {}
            model_device = next(self.parameters()).device
            relobralo_kwargs['device'] = model_device
            relobralo_kwargs['n_losses'] = 2
            self.relobralo = ReLoBRaLoWeights(**relobralo_kwargs)

    def update_curriculum_weights(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if not self.use_curriculum:
            return
        _, _, laplacian_u = self.compute_laplacian(x, y)
        f = self.source_term(x, y)
        pde_residual = laplacian_u - f  
        # Compute weights based on current training progress tau
        self.sample_weights = self.curriculum_weight.compute_weights(pde_residual, self.current_tau)

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        if total_steps == 0:
            self.current_tau = 0.0
        else:
            # Adapt tau saturation point for GMM
            tau_base = total_steps * self.curriculum_weight.tau_saturation
            self.current_tau = min(current_step / tau_base, 1.0)  # Limit tau ≤ 1

    # Forward propagation
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, y], dim=1)
        return self.layers(input_tensor)

    def compute_laplacian(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.forward(x, y)  
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
        u_y = torch.autograd.grad(
            outputs=u,
            inputs=y,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        u_yy = torch.autograd.grad(
            outputs=u_y,
            inputs=y,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        laplacian_u = u_xx + u_yy
        
        return u_xx, u_yy, laplacian_u

    # Define source term f(x,y)
    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sin_beta1pi_x = torch.sin(BETA1 * np.pi * x)
        sin_beta2pi_y = torch.sin(BETA2 * np.pi * y)
        exp_term = torch.exp(-x**2 - y**2)
        laplacian_first = - (np.pi**2) * (BETA1**2 + BETA2**2) * sin_beta1pi_x * sin_beta2pi_y
        laplacian_second = 4 * (x**2 + y**2 - 1) * exp_term
        f = laplacian_first + laplacian_second
        
        return f

    def pde_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, laplacian_u = self.compute_laplacian(x, y)
        f = self.source_term(x, y)
        pde_residual = laplacian_u - f  

        if self.use_curriculum and self.sample_weights is not None:
            # Weighted loss: higher weights for simpler samples initially, gradually decreasing
            weighted_residual = self.sample_weights.detach() * (pde_residual ** 2)
            pde_loss_val = torch.mean(weighted_residual)
        else:
            pde_loss_val = torch.mean(pde_residual ** 2)
        
        # Cache PDE loss value
        self.latest_pde_loss = pde_loss_val.item()
        
        return pde_loss_val
    
    def boundary_loss(self, boundary_data) -> torch.Tensor:
        x0, y0, x1, y1, x2, y2, x3, y3 = boundary_data
        
        u0_pred = self.forward(x0, y0)
        u0_exact = self.analytical_solution(x0, y0)
        
        u1_pred = self.forward(x1, y1)
        u1_exact = self.analytical_solution(x1, y1)
        
        u2_pred = self.forward(x2, y2)
        u2_exact = self.analytical_solution(x2, y2)
        
        u3_pred = self.forward(x3, y3)
        u3_exact = self.analytical_solution(x3, y3)
        
        boundary_residual = (
            torch.mean((u0_pred - u0_exact)**2) +
            torch.mean((u1_pred - u1_exact)**2) +
            torch.mean((u2_pred - u2_exact)**2) +
            torch.mean((u3_pred - u3_exact)**2)
        )
        
        self.latest_boundary_loss = boundary_residual.item()
        
        return boundary_residual
    
    # Analytical solution function
    def analytical_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sin_part = torch.sin(BETA1 * np.pi * x) * torch.sin(BETA2 * np.pi * y)
        exp_part = torch.exp(-x**2 - y**2)
        return sin_part + exp_part
    
    def compute_weighted_total_loss(self, pde_data, boundary_data) -> torch.Tensor:
        x_pde, y_pde = pde_data
        pde_loss_val = self.pde_loss(x_pde, y_pde)
        boundary_loss_val = self.boundary_loss(boundary_data)
        
        # If ReLoBRaLo is enabled, update weights and compute weighted total loss
        if self.use_relobralo:
            current_losses = [self.latest_pde_loss, self.latest_boundary_loss]
            self.relobralo.update(current_losses)
            weights = self.relobralo.weights.detach()
            return weights[0] * pde_loss_val + weights[1] * boundary_loss_val
        else:
            return pde_loss_val + boundary_loss_val
    
    def final_total_loss(self, pde_data, boundary_data) -> tuple[float, float, float]:
        x_pde, y_pde = pde_data
        
        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde, y_pde).item()
            boundary_loss_val = self.boundary_loss(boundary_data).item()
            
            if self.use_relobralo:
                # If ReLoBRaLo is enabled, return weighted total loss
                weights = self.relobralo.weights.detach().cpu().numpy()
                total_loss_val = weights[0] * pde_loss_val + weights[1] * boundary_loss_val
            else:
                total_loss_val = pde_loss_val + boundary_loss_val
        
        return total_loss_val, pde_loss_val, boundary_loss_val

# 2. Generate training and testing data
def generate_data(
    n_pde: int,
    n_boundary: int,
    n_test: int 
) -> tuple:
    x_pde = torch.rand(n_pde, 1) * (DOMAIN_MAX - DOMAIN_MIN) + DOMAIN_MIN
    y_pde = torch.rand(n_pde, 1) * (DOMAIN_MAX - DOMAIN_MIN) + DOMAIN_MIN
    x_pde.requires_grad_(True)
    y_pde.requires_grad_(True)
    pde_data = (x_pde, y_pde)

    x0 = torch.zeros(n_boundary, 1)
    y0 = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_boundary).reshape(-1, 1)
    x1 = torch.ones(n_boundary, 1)
    y1 = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_boundary).reshape(-1, 1)
    x2 = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_boundary).reshape(-1, 1)
    y2 = torch.zeros(n_boundary, 1)
    x3 = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_boundary).reshape(-1, 1)
    y3 = torch.ones(n_boundary, 1)
    
    boundary_data = (x0, y0, x1, y1, x2, y2, x3, y3)
    
    x_test = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_test).reshape(-1, 1)
    y_test = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_test).reshape(-1, 1)
    x_test_grid, y_test_grid = torch.meshgrid(x_test.squeeze(), y_test.squeeze(), indexing='ij')
    x_test_flat = x_test_grid.reshape(-1, 1)
    y_test_flat = y_test_grid.reshape(-1, 1)
    test_data = (x_test_flat, y_test_flat)
    
    return pde_data, boundary_data, test_data

# 3. Training function (Adam)
def train_with_optimizer(
    model: CGMPINN2D,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    step_offset: int = 0,           # Step offset
    global_total_steps: int = None  # Global total steps
) -> int:  # Return cumulative steps
    x_pde, y_pde = pde_data
    
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
            model.update_curriculum_weights(x_pde, y_pde)
            print(f"  → Epoch {epoch+1} (Global step {global_step}), curriculum weights (tau={model.current_tau:.3f})")

        # Loss calculation and backpropagation (using ReLoBRaLo weighted total loss)
        total_loss = model.compute_weighted_total_loss(pde_data, boundary_data)
        
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

# 4. Training function (L-BFGS)
def train_with_lbfgs(
    model: CGMPINN2D,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    lr: float,
    max_iter: int,
    step_offset: int = 0,           # Step offset
    global_total_steps: int = None  # Global total steps
) -> int:  # Return cumulative steps
    x_pde, y_pde = pde_data
    
    model.train()

    # If global total steps not specified, use current phase's max_iter
    if global_total_steps is None:
        global_total_steps = max_iter

    pde_loss_val = boundary_loss_val = 0.0

    # Initial curriculum weights update
    if model.use_curriculum:
        model.update_curriculum_weights(x_pde, y_pde)

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, boundary_loss_val
        optimizer.zero_grad()
        total_loss = model.compute_weighted_total_loss(pde_data, boundary_data)
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

        # Periodically update curriculum weights
        if model.use_curriculum and (iter_idx % model.curriculum_weight.update_interval == 0):
            model.update_curriculum_weights(x_pde, y_pde)
            print(f"  → Iteration {iter_idx+1} (global step {global_step}), current curriculum weight (tau={model.current_tau:.3f})")

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

# 5. Two-stage training function (Adam→L-BFGS)
def train_adam_lbfgs(
    model: CGMPINN2D,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    # Calculate global total steps (Adam + L-BFGS)
    global_total_steps = adam_epochs + lbfgs_max_iter
    
    print(f"\n=== Stage 1: Adam Global Exploration (Starting from simple samples) ===")
    print(f"    Global step range: 1 ~ {adam_epochs}, tau: 0 → {adam_epochs/global_total_steps:.3f}")
    
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    current_step = train_with_optimizer(
        model=model, 
        optimizer=optimizer_adam, 
        epochs=adam_epochs,
        pde_data=pde_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        step_offset=0,                        # Adam starts from step 0
        global_total_steps=global_total_steps # Use global total steps
    )
    
    print(f"\n=== Stage 2: L-BFGS Local Fine-tuning (Gradually focusing on difficult samples) ===")
    print(f"    Global step range: {current_step+1} ~ {global_total_steps}, tau: {current_step/global_total_steps:.3f} → 1.0")
    
    train_with_lbfgs(
        model=model, 
        pde_data=pde_data,
        boundary_data=boundary_data,
        loss_history=loss_history, 
        lr=lr_lbfgs, 
        max_iter=lbfgs_max_iter,
        step_offset=current_step,             # L-BFGS continues from where Adam left off
        global_total_steps=global_total_steps # Use global total steps
    )

# 6. Evaluation function (with multiple error metrics)
def evaluate_model(
    model: CGMPINN2D,
    test_data: tuple,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    x_test, y_test = test_data
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test, y_test)
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        u_exact_l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()
        l2_relative_error = l2_error / u_exact_l2_norm if u_exact_l2_norm > 1e-10 else float('inf')
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        pointwise_error = torch.abs(u_pred - u_exact).numpy()
        u_pred_np = u_pred.numpy()
    
    return l2_error, l2_relative_error, linf_error, pointwise_error, u_pred_np

# 7. Visualization functions
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
    x_test, y_test = test_data
    n_test = int(np.sqrt(len(x_test)))
    
    # Reshape grids
    x_grid = x_test.reshape(n_test, n_test).numpy()
    y_grid = y_test.reshape(n_test, n_test).numpy()
    u_exact_grid = u_exact_np.reshape(n_test, n_test)
    
    for activation_name, results in activation_results.items():
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        fig.suptitle(f'Solution Comparison - Activation: {activation_name}', fontsize=24, y=0.99)
        
        for idx, optim_label in enumerate(optim_labels):
            u_pred = results[optim_label]['u_pred']
            pointwise_error = results[optim_label]['pointwise_error']
            
            # Reshape to grids
            u_pred_grid = u_pred.reshape(n_test, n_test)
            error_grid = pointwise_error.reshape(n_test, n_test)
            
            # Predicted solution
            ax1 = axes[idx, 0]
            im1 = ax1.pcolormesh(x_grid, y_grid, u_pred_grid, cmap=cm.jet, shading='gouraud')
            ax1.set_xlabel('x', fontsize=10)
            ax1.set_ylabel('y', fontsize=10)
            ax1.set_title(f'{optim_label} - Predicted', fontsize=12)
            plt.colorbar(im1, ax=ax1, shrink=0.6, aspect=5)
            
            # Analytical solution
            ax2 = axes[idx, 1]
            im2 = ax2.pcolormesh(x_grid, y_grid, u_exact_grid, cmap=cm.jet, shading='gouraud')
            ax2.set_title('Analytical Solution', fontsize=12)
            plt.colorbar(im2, ax=ax2, shrink=0.6, aspect=5)
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('y', fontsize=10)
            
            # Absolute error
            ax3 = axes[idx, 2]
            im3 = ax3.pcolormesh(x_grid, y_grid, error_grid, cmap=cm.viridis, shading='gouraud')
            ax3.set_xlabel('x', fontsize=10)
            ax3.set_ylabel('y', fontsize=10)
            ax3.set_title(f'{optim_label} - Absolute Error', fontsize=12)
            plt.colorbar(im3, ax=ax3, shrink=0.6, aspect=5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

# 8. Main function (Execution flow)
def train_2D_poisson_CGMPINN_with_activations_and_source_term():
    # 1. Hyperparameter settings
    n_layers = 4              
    input_dim = 2             
    output_dim = 1            
    hidden_dim = 50           
    n_pde = 2000             
    n_boundary = 250          
    n_test = 100              
    lr_lbfgs_config = {'Tanh': 1.0}
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
        'n_losses': 2,
        'alpha': 0.999,
        'temperature': 1.0,
        'rho': 0.99,
        'device': 'cpu'
    }
    
    # Activation functions dictionary
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (CGMPINN)', 'L-BFGS (CGMPINN)', 'Adam→L-BFGS (CGMPINN)']
    
    # 2. Generate data 
    print("Generating training and testing data...")
    pde_data, boundary_data, test_data = generate_data(
        n_pde=n_pde,
        n_boundary=n_boundary,
        n_test=n_test
    )
    x_test, y_test = test_data
    
    # Compute exact solution
    model_temp = CGMPINN2D(input_dim, output_dim, hidden_dim, 1, TanhActivation())
    u_exact = model_temp.analytical_solution(x_test, y_test)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name} | Strategy: GMM Curriculum Learning + ReLoBRaLo Adaptive Weight")
        print("="*80)
        
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Adam training (CGMPINN + ReLoBRaLo + GMM)
        print(f"\n--- {activation_name} + Adam (CGMPINN+ReLoBRaLo) ---")
        model_adam = CGMPINN2D(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs # ReLoBRaLo optionally enabled
        )
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        start_time = time.time()
        train_with_optimizer(
            model=model_adam,
            optimizer=optimizer_adam,
            epochs=epochs_adam,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam.final_total_loss(pde_data, boundary_data)
        current_results['Adam (CGMPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred,
            'pointwise_error': pointwise_err
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 L-BFGS training (CGMPINN + GMM)
        print(f"\n--- {activation_name} + L-BFGS (CGMPINN) ---")
        model_lbfgs = CGMPINN2D(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=False, relobralo_kwargs=relobralo_kwargs
        )
        loss_history_lbfgs = []
        start_time = time.time()
        train_with_lbfgs(
            model=model_lbfgs,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_lbfgs,
            lr=lr_lbfgs_config[activation_name],
            max_iter=max_iter_lbfgs
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate L-BFGS
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_lbfgs.final_total_loss(pde_data, boundary_data)
        current_results['L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred,
            'pointwise_error': pointwise_err
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Adam→L-BFGS training (CGMPINN + ReLoBRaLo + GMM)
        print(f"\n--- {activation_name} + Adam→L-BFGS (CGMPINN+ReLoBRaLo) ---")
        model_adam_lbfgs = CGMPINN2D(
            input_dim, output_dim, hidden_dim, n_layers,
            activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
            use_relobralo=True, relobralo_kwargs=relobralo_kwargs # ReLoBRaLo optionally enabled
        )
        loss_history_adam_lbfgs = []
        start_time = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs,
            lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam→L-BFGS
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam_lbfgs.final_total_loss(pde_data, boundary_data)
        current_results['Adam→L-BFGS (CGMPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred,
            'pointwise_error': pointwise_err
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        
        # save complete information for Tanh + Adam→L-BFGS
        if activation_name == 'Tanh':  
            save_path = "2D_poisson_tanh_adam2lbfgs_cgmpinn.pth"
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
            print(f"✅ Tanh+Adam→LBFGS complete information saved to: {save_path}")
        
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save current activation function results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # 5. Results Visualization
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # 6. Print Comprehensive Comparison Table
    print("\n" + "="*140)
    print("Comprehensive Comparison Table for All Activation Functions × GMM + Curriculum Learning + ReLoBRaLo Adaptive Weights × Optimizers (2D Poisson Equation)")
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
    
    # 7. Print Best Performance for Each Activation Function
    print("\n" + "="*110)
    print("Best Performance for Each Activation Function (GMM + Curriculum Learning + ReLoBRaLo) Sorted by Standard L2 Error")
    print("="*110)
    for activation_name in activation_names:
        print(f"\n【{activation_name} - GMM + Curriculum Learning + ReLoBRaLo Adaptive Weights】")
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
train_2D_poisson_CGMPINN_with_activations_and_source_term()