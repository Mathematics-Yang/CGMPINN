import numpy as np
import os
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

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

class TanhActivation(nn.Module):
    """Helper for the ablation experiment."""
    def forward(self, x):
        return torch.tanh(x)

BETA1 = 3.0
BETA2 = 2.0
DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0

class AdaptiveLossWeights:
    """Helper for the ablation experiment."""
    def __init__(self, n_losses=2, device='cpu'):
        super().__init__()
        self.n_losses = n_losses
        self.device = device
        self.weights = torch.ones(n_losses, device=device, dtype=torch.float32)

class ReLoBRaLoWeights(AdaptiveLossWeights):
    """Helper for the ablation experiment."""
    def __init__(self, n_losses=2, alpha=0.999, temperature=1.0, 
                 rho=0.99, device='cpu'):
        super().__init__(n_losses, device)
        self.alpha = alpha
        self.temperature = temperature
        self.rho = rho
        
        self.loss_history = [[] for _ in range(n_losses)]
        self.ema_losses = None
        self.initial_losses = None
        
    def update(self, losses, **kwargs):
        """Helper for the ablation experiment."""
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
        """Helper for the ablation experiment."""
        self.n_components = n_components
        self.update_interval = update_interval
        self.epsilon = epsilon
        self.beta = beta
        self.tau_saturation = tau_saturation
        self.use_variance_factor = use_variance_factor
        self.gmm = GaussianMixture(n_components=n_components, random_state=1224)
        self._last_valid_weights = None

    def _normalize_point_weights(self, point_weights: np.ndarray) -> torch.Tensor:
        point_weights = point_weights / (point_weights.mean() + self.epsilon)
        return torch.tensor(point_weights.reshape(-1, 1), dtype=torch.float32)

    def compute_gmm_weights(self, residuals: torch.Tensor) -> torch.Tensor:
        res_np = residuals.detach().cpu().numpy().reshape(-1, 1)
        try:
            self.gmm.fit(res_np)
            gamma = self.gmm.predict_proba(res_np)
            sigma_sq = self.gmm.covariances_.flatten()
            res_squared = res_np.flatten() ** 2
            component_difficulty = np.array([
                np.sum(gamma[:, j] * res_squared) / (np.sum(gamma[:, j]) + self.epsilon)
                for j in range(self.n_components)
            ])
            diff_min, diff_max = component_difficulty.min(), component_difficulty.max()
            if diff_max - diff_min > self.epsilon:
                normalized_diff = (component_difficulty - diff_min) / (diff_max - diff_min)
            else:
                normalized_diff = np.zeros_like(component_difficulty)
            component_weights = np.exp(self.beta * normalized_diff)
            if self.use_variance_factor:
                variance_factor = 1 / (sigma_sq + self.epsilon)
                variance_factor = variance_factor / (variance_factor.max() + self.epsilon)
                component_weights = component_weights * variance_factor
            point_weights = np.sum(gamma * component_weights[np.newaxis, :], axis=1)
            self._last_valid_weights = point_weights.copy()
        except Exception as e:
            print(f"GMM training failed: {e}")
            if self._last_valid_weights is not None and len(self._last_valid_weights) == len(res_np):
                point_weights = self._last_valid_weights
            else:
                point_weights = np.ones(len(res_np))
        return self._normalize_point_weights(point_weights).to(residuals.device)

    def compute_cl_weights(self, residuals: torch.Tensor, tau: float) -> torch.Tensor:
        res_np = residuals.detach().cpu().numpy().reshape(-1)
        point_difficulty = res_np ** 2
        diff_min, diff_max = point_difficulty.min(), point_difficulty.max()
        if diff_max - diff_min > self.epsilon:
            normalized_diff = (point_difficulty - diff_min) / (diff_max - diff_min)
        else:
            normalized_diff = np.zeros_like(point_difficulty)
        easy_weight = np.exp(-self.beta * normalized_diff)
        hard_weight = np.exp(-self.beta * (1 - normalized_diff))
        point_weights = (1 - tau) * easy_weight + tau * hard_weight
        return self._normalize_point_weights(point_weights).to(residuals.device)

    def compute_weights(self, residuals: torch.Tensor, tau: float, mode: str = 'cgm') -> torch.Tensor:
        if mode == 'gmm':
            return self.compute_gmm_weights(residuals)
        if mode == 'cl':
            return self.compute_cl_weights(residuals, tau)
        """Helper for the ablation experiment."""
        res_np = residuals.detach().cpu().numpy().reshape(-1, 1)
        
        try:
            self.gmm.fit(res_np)
            gamma = self.gmm.predict_proba(res_np)
            sigma_sq = self.gmm.covariances_.flatten()
            
            res_squared = res_np.flatten() ** 2
            component_difficulty = np.array([
                np.sum(gamma[:, j] * res_squared) / (np.sum(gamma[:, j]) + self.epsilon)
                for j in range(self.n_components)
            ])
            
            diff_min, diff_max = component_difficulty.min(), component_difficulty.max()
            if diff_max - diff_min > self.epsilon:
                normalized_diff = (component_difficulty - diff_min) / (diff_max - diff_min)
            else:
                normalized_diff = np.zeros_like(component_difficulty)
            
            easy_weight = np.exp(-self.beta * normalized_diff)
            hard_weight = np.exp(-self.beta * (1 - normalized_diff))
            
            curriculum_weight = (1 - tau) * easy_weight + tau * hard_weight
            
            if self.use_variance_factor:
                variance_factor = 1 / (sigma_sq + self.epsilon)
                variance_factor = variance_factor / variance_factor.max()
                effective_variance_factor = (1 - tau) * variance_factor + tau * 1.0
                component_weights = curriculum_weight * effective_variance_factor
            else:
                component_weights = curriculum_weight
            
            point_weights = np.sum(gamma * component_weights[np.newaxis, :], axis=1)
            self._last_valid_weights = point_weights.copy()
        
        except Exception as e:
            print(f"GMM training failed: {e}")
            if self._last_valid_weights is not None and len(self._last_valid_weights) == len(res_np):
                point_weights = self._last_valid_weights
            else:
                point_weights = np.ones(len(res_np))
        
        point_weights = point_weights / (point_weights.mean() + self.epsilon)
        
        return torch.tensor(point_weights.reshape(-1, 1), dtype=torch.float32).to(residuals.device)

class CGMPINN2D(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, 
                 use_curriculum=True, gmm_kwargs=None,
                 use_relobralo=True, relobralo_kwargs=None,
                 weight_strategy='cgm'):
        super(CGMPINN2D, self).__init__()
        self.activation = activation
        self.weight_strategy = weight_strategy.lower()
        self.use_curriculum = use_curriculum and self.weight_strategy in {'gmm', 'cl', 'cgm'}
        self.use_relobralo = use_relobralo
        self.total_train_steps = 0
        self.current_tau = 0.0

        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

        if self.use_curriculum:
            gmm_kwargs = gmm_kwargs or {}
            self.curriculum_weight = GMMCurriculumWeight(**gmm_kwargs)
            self.sample_weights = None
        
        self.latest_pde_loss = None
        self.latest_boundary_loss = None

        if self.use_relobralo:
            relobralo_kwargs = relobralo_kwargs or {}
            model_device = next(self.parameters()).device
            relobralo_kwargs['device'] = model_device
            relobralo_kwargs['n_losses'] = 2
            self.relobralo = ReLoBRaLoWeights(**relobralo_kwargs)

    def update_curriculum_weights(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Helper for the ablation experiment."""
        if not self.use_curriculum:
            return
        _, _, laplacian_u = self.compute_laplacian(x, y)
        f = self.source_term(x, y)
        pde_residual = laplacian_u - f
        self.sample_weights = self.curriculum_weight.compute_weights(
            pde_residual, self.current_tau, mode=self.weight_strategy
        )

    def set_training_progress(self, current_step: int, total_steps: int) -> None:
        """Helper for the ablation experiment."""
        if total_steps == 0:
            self.current_tau = 0.0
        else:
            tau_base = total_steps * self.curriculum_weight.tau_saturation if self.use_curriculum else total_steps
            self.current_tau = min(current_step / tau_base, 1.0)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Helper for the ablation experiment."""
        input_tensor = torch.cat([x, y], dim=1)
        return self.layers(input_tensor)

    def compute_laplacian(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper for the ablation experiment."""
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

    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Helper for the ablation experiment."""
        sin_beta1pi_x = torch.sin(BETA1 * np.pi * x)
        sin_beta2pi_y = torch.sin(BETA2 * np.pi * y)
        exp_term = torch.exp(-x**2 - y**2)
        
        laplacian_first = - (np.pi**2) * (BETA1**2 + BETA2**2) * sin_beta1pi_x * sin_beta2pi_y
        
        laplacian_second = 4 * (x**2 + y**2 - 1) * exp_term
        
        f = laplacian_first + laplacian_second
        
        return f

    def pde_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Helper for the ablation experiment."""
        _, _, laplacian_u = self.compute_laplacian(x, y)
        f = self.source_term(x, y)
        pde_residual = laplacian_u - f

        if self.use_curriculum and self.sample_weights is not None:
            weighted_residual = self.sample_weights.detach() * (pde_residual ** 2)
            pde_loss_val = torch.mean(weighted_residual)
        else:
            pde_loss_val = torch.mean(pde_residual ** 2)
        
        self.latest_pde_loss = pde_loss_val.item()
        
        return pde_loss_val
    
    def boundary_loss(self, boundary_data) -> torch.Tensor:
        """Helper for the ablation experiment."""
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
    
    def analytical_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Helper for the ablation experiment."""
        sin_part = torch.sin(BETA1 * np.pi * x) * torch.sin(BETA2 * np.pi * y)
        exp_part = torch.exp(-x**2 - y**2)
        return sin_part + exp_part
    
    def compute_weighted_total_loss(self, pde_data, boundary_data) -> torch.Tensor:
        """Helper for the ablation experiment."""
        x_pde, y_pde = pde_data
        
        pde_loss_val = self.pde_loss(x_pde, y_pde)
        boundary_loss_val = self.boundary_loss(boundary_data)
        
        if self.use_relobralo:
            current_losses = [self.latest_pde_loss, self.latest_boundary_loss]
            self.relobralo.update(current_losses)
            weights = self.relobralo.weights.detach()
            return weights[0] * pde_loss_val + weights[1] * boundary_loss_val
        else:
            return pde_loss_val + boundary_loss_val
    
    def final_total_loss(self, pde_data, boundary_data) -> tuple[float, float, float]:
        """Helper for the ablation experiment."""
        x_pde, y_pde = pde_data
        
        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde, y_pde).item()
            boundary_loss_val = self.boundary_loss(boundary_data).item()
            
            if self.use_relobralo:
                weights = self.relobralo.weights.detach().cpu().numpy()
                total_loss_val = weights[0] * pde_loss_val + weights[1] * boundary_loss_val
            else:
                total_loss_val = pde_loss_val + boundary_loss_val
        
        return total_loss_val, pde_loss_val, boundary_loss_val

def generate_data(
    n_pde: int,
    n_boundary: int,
    n_test: int 
) -> tuple:
    """Helper for the ablation experiment."""
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

def train_with_optimizer(
    model: CGMPINN2D,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    step_offset: int = 0,
    global_total_steps: int = None
) -> int:
    """Helper for the ablation experiment."""
    x_pde, y_pde = pde_data
    
    model.train()

    if global_total_steps is None:
        global_total_steps = epochs

    for epoch in range(epochs):
        global_step = step_offset + epoch + 1
        
        model.set_training_progress(global_step, global_total_steps)
        
        if model.use_curriculum and epoch % model.curriculum_weight.update_interval == 0:
            model.update_curriculum_weights(x_pde, y_pde)
            print(f"  -> Epoch {epoch+1} (global step {global_step}), curriculum tau={model.current_tau:.3f}")

        total_loss = model.compute_weighted_total_loss(pde_data, boundary_data)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if (epoch + 1) % 1000 == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            relobralo_info = ""

            pde_loss = model.latest_pde_loss if model.latest_pde_loss is not None else 0.0
            boundary_loss = model.latest_boundary_loss if model.latest_boundary_loss is not None else 0.0

            if model.use_relobralo:
                weights = model.relobralo.weights.detach().cpu().numpy()
                relobralo_info = f" | ReLoBRaLo weights (PDE/BC): [{weights[0]:.2e}, {weights[1]:.2e}]"
            print(
                f'Epoch {epoch+1:4d}{curr_info}{relobralo_info} | Total loss: {total_loss.item():.2e} | '
                f'PDE loss: {pde_loss:.2e} | Boundary loss: {boundary_loss:.2e}'
            )
    
    return step_offset + epochs

def train_with_lbfgs(
    model: CGMPINN2D,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    lr: float,
    max_iter: int,
    step_offset: int = 0,
    global_total_steps: int = None
) -> int:
    """Helper for the ablation experiment."""
    x_pde, y_pde = pde_data
    
    model.train()

    if global_total_steps is None:
        global_total_steps = max_iter

    pde_loss_val = boundary_loss_val = 0.0

    if model.use_curriculum:
        model.update_curriculum_weights(x_pde, y_pde)

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, boundary_loss_val
        optimizer.zero_grad()
        total_loss = model.compute_weighted_total_loss(pde_data, boundary_data)
        total_loss.backward()
        
        pde_loss_val = model.latest_pde_loss if model.latest_pde_loss is not None else 0.0
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
            model.update_curriculum_weights(x_pde, y_pde)
            print(f"  -> Iteration {iter_idx+1} (global step {global_step}), curriculum tau={model.current_tau:.3f}")

        loss_history.append(total_loss.item())

        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            curr_info = f" (tau={model.current_tau:.3f})" if model.use_curriculum else ""
            relobralo_info = ""
            if model.use_relobralo:
                weights = model.relobralo.weights.detach().cpu().numpy()
                relobralo_info = f" | ReLoBRaLo weights (PDE/BC): [{weights[0]:.2e}, {weights[1]:.2e}]"
            print(
                f'Iteration {iter_idx+1}/{max_iter}{curr_info}{relobralo_info} | Total loss: {total_loss.item():.2e} | '
                f'PDE loss: {pde_loss_val:.2e} | Boundary loss: {boundary_loss_val:.2e}'
            )
    
    return step_offset + max_iter

def train_adam_lbfgs(
    model: CGMPINN2D,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    """Helper for the ablation experiment."""
    global_total_steps = adam_epochs + lbfgs_max_iter
    
    print("\n=== Stage 1: Adam global exploration (starting from easy samples) ===")
    print(f"    Global step range: 1 ~ {adam_epochs}, tau: 0 -> {adam_epochs/global_total_steps:.3f}")
    
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    current_step = train_with_optimizer(
        model=model, 
        optimizer=optimizer_adam, 
        epochs=adam_epochs,
        pde_data=pde_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        step_offset=0,
        global_total_steps=global_total_steps
    )
    
    print("\n=== Stage 2: L-BFGS local refinement (gradually focusing on hard samples) ===")
    print(f"    Global step range: {current_step+1} ~ {global_total_steps}, tau: {current_step/global_total_steps:.3f} -> 1.0")
    
    train_with_lbfgs(
        model=model, 
        pde_data=pde_data,
        boundary_data=boundary_data,
        loss_history=loss_history, 
        lr=lr_lbfgs, 
        max_iter=lbfgs_max_iter,
        step_offset=current_step,
        global_total_steps=global_total_steps
    )

def get_script_dir(problem_name: str) -> str:
    if '__file__' in globals():
        return os.path.dirname(os.path.abspath(__file__))

    target_file = f"{problem_name}_ablation.py"
    current_dir = os.getcwd()
    while True:
        matched_dir = None
        for root, _, files in os.walk(current_dir):
            if target_file not in files:
                continue
            if os.path.basename(root) == f"{problem_name}_code":
                return root
            if matched_dir is None:
                matched_dir = root
        if matched_dir is not None:
            return matched_dir

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return os.getcwd()
        current_dir = parent_dir


def build_save_path(problem_name: str, activation_name: str, method_name: str) -> str:
    suffix_map = {'GMM': 'gmmpinn', 'CL': 'clpinn', 'CGM': 'cgmpinn'}
    filename = f"{problem_name}_{activation_name.lower()}_adam2lbfgs_{suffix_map[method_name]}.pth"
    current_script_dir = get_script_dir(problem_name)
    parameter_dir = os.path.join(current_script_dir, "..", "2D_poisson_parameter")
    os.makedirs(parameter_dir, exist_ok=True)
    return os.path.join(parameter_dir, filename)


def save_training_artifacts(
    save_path: str,
    model,
    loss_history: list,
    final_total_loss: float,
    training_time: float,
    activation_name: str,
    weight_strategy: str,
    hyper_parameters: dict
) -> None:
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'loss_history': loss_history,
            'final_total_loss': final_total_loss,
            'training_time': training_time,
            'activation_name': activation_name,
            'weight_strategy': weight_strategy,
            'hyper_parameters': hyper_parameters,
        },
        save_path
    )


def evaluate_model(
    model: CGMPINN2D,
    test_data: tuple,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Helper for the ablation experiment."""
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

def plot_combined(activation_loss_histories: dict, activation_results: dict,
                  test_data: tuple, u_exact_np: np.ndarray, optim_labels: list) -> None:
    """Helper for the ablation experiment."""
    x_test, y_test = test_data
    n_test = int(np.sqrt(len(x_test)))
    x_grid = x_test.reshape(n_test, n_test).numpy()
    y_grid = y_test.reshape(n_test, n_test).numpy()
    u_exact_grid = u_exact_np.reshape(n_test, n_test)
    
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    ax_loss, ax_pred, ax_exact, ax_error = axes
    
    color_palette = {
        'optimizers': ['#e74c3c', '#3498db', '#27ae60'],
    }
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    activation_name = list(activation_loss_histories.keys())[0]
    loss_histories = activation_loss_histories[activation_name]
    for opt_idx, (loss_history, label) in enumerate(zip(loss_histories, optim_labels)):
        marker_interval = max(1, len(loss_history) // 8)
        ax_loss.plot(loss_history, label=f'{label}', 
                     color=color_palette['optimizers'][opt_idx], linestyle=line_styles[opt_idx],
                     marker=markers[opt_idx], markevery=marker_interval, markersize=6,
                     linewidth=2, alpha=0.9)
    ax_loss.set_xlabel('Iteration', fontsize=12)
    ax_loss.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax_loss.set_title(f'(a) Optimizer Training Loss Comparison', pad=10, fontsize=14)
    ax_loss.set_yscale('log')
    ax_loss.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_loss.tick_params(axis='both', labelsize=10)
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)
    
    adam2lbfgs_res = activation_results[activation_name]['CGM']
    u_pred = adam2lbfgs_res['u_pred']
    pointwise_error = adam2lbfgs_res['pointwise_error']
    u_pred_grid = u_pred.reshape(n_test, n_test)
    error_grid = pointwise_error.reshape(n_test, n_test)
    
    im_pred = ax_pred.pcolormesh(x_grid, y_grid, u_pred_grid, cmap=cm.jet, shading='gouraud')
    ax_pred.set_xlabel('Position $x$', fontsize=12)
    ax_pred.set_ylabel('Position $y$', fontsize=12)
    ax_pred.set_title(f'(b) CGM Prediction', fontsize=14, pad=10)
    plt.colorbar(im_pred, ax=ax_pred, shrink=0.8, aspect=20, pad=0.02)
    ax_pred.tick_params(axis='both', labelsize=10)
    ax_pred.set_aspect('equal', adjustable='box')
    
    im_exact = ax_exact.pcolormesh(x_grid, y_grid, u_exact_grid, cmap=cm.jet, shading='gouraud')
    ax_exact.set_xlabel('Position $x$', fontsize=12)
    ax_exact.set_ylabel('Position $y$', fontsize=12)
    ax_exact.set_title('(c) Exact Solution', fontsize=14, pad=10)
    plt.colorbar(im_exact, ax=ax_exact, shrink=0.8, aspect=20, pad=0.02)
    ax_exact.tick_params(axis='both', labelsize=10)
    ax_exact.set_aspect('equal', adjustable='box')
    
    im_error = ax_error.pcolormesh(x_grid, y_grid, error_grid, cmap=cm.RdBu_r, shading='gouraud')
    ax_error.set_xlabel('Position $x$', fontsize=12)
    ax_error.set_ylabel('Position $y$', fontsize=12)
    ax_error.set_title('(d) Absolute Error Distribution', fontsize=14, pad=10)
    plt.colorbar(im_error, ax=ax_error, shrink=0.8, aspect=20, pad=0.02)
    ax_error.tick_params(axis='both', labelsize=10)
    ax_error.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace=0.20)
    plt.show()

def run_2D_poisson_ablation():
    n_layers = 4
    input_dim = 2
    output_dim = 1
    hidden_dim = 50
    n_pde = 2000
    n_boundary = 250
    n_test = 100
    lr_adam_lbfgs = 0.8
    adam_lbfgs_adam_epochs = 5000
    adam_lbfgs_lbfgs_iter = 10000

    gmm_kwargs = {
        'n_components': 4,
        'update_interval': 200,
        'epsilon': 1e-6,
        'beta': 1.0,
        'tau_saturation': 0.8,
        'use_variance_factor': True
    }
    relobralo_kwargs = {
        'n_losses': 2,
        'alpha': 0.999,
        'temperature': 1.0,
        'rho': 0.99,
        'device': 'cpu'
    }

    activations = {'Tanh': TanhActivation()}
    method_labels = ['GMM', 'CL', 'CGM']
    problem_name = "2D_poisson"
    use_relobralo_for_two_stage = True

    pde_data, boundary_data, test_data = generate_data(n_pde=n_pde, n_boundary=n_boundary, n_test=n_test)
    x_test, y_test = test_data
    model_temp = CGMPINN2D(input_dim, output_dim, hidden_dim, 1, TanhActivation())
    u_exact = model_temp.analytical_solution(x_test, y_test)
    u_exact_np = u_exact.numpy()

    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}

    for activation_name, activation_fn in activations.items():
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        for method_name in method_labels:
            weight_strategy = method_name.lower()
            save_path = build_save_path(problem_name, activation_name, method_name)
            model = CGMPINN2D(
                input_dim, output_dim, hidden_dim, n_layers,
                activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
                use_relobralo=use_relobralo_for_two_stage, relobralo_kwargs=relobralo_kwargs,
                weight_strategy=weight_strategy
            )
            if method_name == 'CGM' or os.path.exists(save_path):
                save_dict = torch.load(save_path, map_location='cpu', weights_only=False)
                model.load_state_dict(save_dict['model_state_dict'])
                loss_history = save_dict['loss_history']
                training_time = save_dict['training_time']
            else:
                loss_history = []
                start_time = time.time()
                train_adam_lbfgs(
                    model=model, pde_data=pde_data, boundary_data=boundary_data,
                    loss_history=loss_history, lr_lbfgs=lr_adam_lbfgs,
                    adam_epochs=adam_lbfgs_adam_epochs, lbfgs_max_iter=adam_lbfgs_lbfgs_iter
                )
                training_time = time.time() - start_time
                save_training_artifacts(
                    save_path=save_path,
                    model=model,
                    loss_history=loss_history,
                    final_total_loss=model.final_total_loss(pde_data, boundary_data)[0],
                    training_time=training_time,
                    activation_name=activation_name,
                    weight_strategy=weight_strategy,
                    hyper_parameters={'n_layers': n_layers, 'hidden_dim': hidden_dim}
                )
            l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model, test_data, u_exact)
            final_total, final_pde, final_boundary = model.final_total_loss(pde_data, boundary_data)
            current_loss_histories.append(loss_history)
            current_training_times.append(training_time)
            current_results[method_name] = {
                'l2_error': l2_err,
                'l2_relative_error': l2_rel_err,
                'linf_error': linf_err,
                'final_total_loss': final_total,
                'final_pde_loss': final_pde,
                'final_boundary_loss': final_boundary,
                'u_pred': u_pred,
                'pointwise_error': pointwise_err
            }
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times

    plot_combined(activation_loss_histories, activation_results, test_data, u_exact_np, method_labels)

    print("\n" + "=" * 140)
    print("2D Poisson Adam->L-BFGS Ablation Comparison Table")
    print("=" * 140)
    print(f"{'Activation':<12} {'Method':<12} {'TrainTime(s)':<14} {'L2Error':<12} {'RelL2Error':<12} {'LinfError':<12} {'FinalTotalLoss':<14}")
    print("-" * 140)
    for activation_name in activations.keys():
        for idx, method_name in enumerate(method_labels):
            results = activation_results[activation_name][method_name]
            train_time = activation_training_times[activation_name][idx]
            print(
                f"{activation_name:<10} {method_name:<12} {train_time:<12.2f} "
                f"{results['l2_error']:<12.2e} {results['l2_relative_error']:<12.2e} "
                f"{results['linf_error']:<12.2e} {results['final_total_loss']:<12.2e} "
            )


run_2D_poisson_ablation()
