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

torch.manual_seed(1224)
np.random.seed(1224)

ALPHA1 = 5.0
ALPHA2 = 3.0
k = 20.0

class TanhActivation(nn.Module):
    """Helper for the ablation experiment."""
    def forward(self, x):
        return torch.tanh(x)

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
                 beta=1.0,
                 tau_saturation=0.8,
                 use_variance_factor=False):
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

class CGMPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, 
                 use_curriculum=True, gmm_kwargs=None,
                 use_relobralo=True, relobralo_kwargs=None,
                 weight_strategy='cgm'):
        super(CGMPINN, self).__init__()
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

    def update_curriculum_weights(self, x: torch.Tensor) -> None:
        """Helper for the ablation experiment."""
        if not self.use_curriculum:
            return
        u_xx = self.compute_gradients(x)
        pde_residual = u_xx - self.source_term(x)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for the ablation experiment."""
        return self.layers(x)
    
    def compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for the ablation experiment."""
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
    
    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for the ablation experiment."""
        return torch.sin(ALPHA1 * np.pi * x) * torch.cos(ALPHA2 * np.pi * x) + torch.tanh(k * x)

    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        """Helper for the ablation experiment."""
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
        """Helper for the ablation experiment."""
        u_xx = self.compute_gradients(x)
        f = self.source_term(x)
        pde_residual = u_xx - f

        if self.use_curriculum and self.sample_weights is not None:
            weighted_residual = self.sample_weights.detach() * (pde_residual ** 2)
            pde_loss_val = torch.mean(weighted_residual)
        else:
            pde_loss_val = torch.mean(pde_residual ** 2)
        
        self.latest_pde_loss = pde_loss_val.item()
        
        return pde_loss_val
    
    def boundary_loss(self) -> torch.Tensor:
        """Helper for the ablation experiment."""
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
        """Helper for the ablation experiment."""
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
    
    def final_total_loss(self, pde_data) -> tuple[float, float, float]:
        """Helper for the ablation experiment."""
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

def generate_data(
    n_pde: int,
    n_test: int 
) -> tuple:
    """Helper for the ablation experiment."""
    x_pde = torch.rand(n_pde, 1, requires_grad=True)
    pde_data = x_pde
    
    x_test = torch.linspace(0.0, 1.0, n_test).reshape(-1, 1)
    test_data = x_test
    
    return pde_data, test_data

def train_with_optimizer(
    model: CGMPINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: torch.Tensor,
    loss_history: list,
    step_offset: int = 0,
    global_total_steps: int = None
) -> int:
    x_pde = pde_data
    model.train()

    if global_total_steps is None:
        global_total_steps = epochs

    for epoch in range(epochs):
        global_step = step_offset + epoch + 1
        
        model.set_training_progress(global_step, global_total_steps)
        
        if model.use_curriculum and epoch % model.curriculum_weight.update_interval == 0:
            model.update_curriculum_weights(x_pde)
            print(f"  -> Epoch {epoch+1} (global step {global_step}), curriculum tau={model.current_tau:.3f}")

        total_loss = model.compute_weighted_total_loss(pde_data)
        
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
    model: CGMPINN,
    pde_data: torch.Tensor,
    loss_history: list,
    lr: float,
    max_iter: int,
    step_offset: int = 0,
    global_total_steps: int = None
) -> int:
    x_pde = pde_data
    model.train()

    if global_total_steps is None:
        global_total_steps = max_iter

    pde_loss_val = boundary_loss_val = 0.0

    if model.use_curriculum:
        model.update_curriculum_weights(x_pde)

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, boundary_loss_val
        optimizer.zero_grad()
        total_loss = model.compute_weighted_total_loss(pde_data)
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
            model.update_curriculum_weights(x_pde)
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
    model: CGMPINN,
    pde_data: torch.Tensor,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    global_total_steps = adam_epochs + lbfgs_max_iter
    
    print("\n=== Stage 1: Adam global exploration (starting from easy samples) ===")
    print(f"    Global step range: 1 ~ {adam_epochs}, tau: 0 -> {adam_epochs/global_total_steps:.3f}")
    
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    current_step = train_with_optimizer(
        model=model, 
        optimizer=optimizer_adam, 
        epochs=adam_epochs,
        pde_data=pde_data,
        loss_history=loss_history,
        step_offset=0,
        global_total_steps=global_total_steps
    )
    
    print("\n=== Stage 2: L-BFGS local refinement (gradually focusing on hard samples) ===")
    print(f"    Global step range: {current_step+1} ~ {global_total_steps}, tau: {current_step/global_total_steps:.3f} -> 1.0")
    
    train_with_lbfgs(
        model=model, 
        pde_data=pde_data,
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
    parameter_dir = os.path.join(current_script_dir, "..", "1D_poisson_parameter")
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
    model: CGMPINN,
    test_data: torch.Tensor,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Helper for the ablation experiment."""
    x_test = test_data
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test)
        
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        
        u_exact_l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()
        l2_relative_error = l2_error / u_exact_l2_norm if u_exact_l2_norm > 1e-10 else float('inf')
        
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        
        pointwise_error = torch.abs(u_pred - u_exact).numpy()
        u_pred_np = u_pred.numpy()
    
    return l2_error, l2_relative_error, linf_error, pointwise_error, u_pred_np


def plot_combined_results(
    activation_loss_histories,
    activation_results,
    test_data,
    u_exact_np,
    optim_labels
) -> None:
    """Helper for the ablation experiment."""
    activation_name = list(activation_loss_histories.keys())[0]
    loss_histories = activation_loss_histories[activation_name]
    results = activation_results[activation_name]
    x_test_np = test_data.numpy().flatten()
    
    color_palette = {
        'exact': '#2c3e50',
        'optimizers': ['#e74c3c', '#3498db', '#27ae60'],
        'error': '#e67e22'
    }
    line_styles = ['-', '--', '-.']
    markers = ['o', 's', '^']
    
    fig, axes = plt.subplots(1, 4, figsize=(32, 6))
    fig.patch.set_facecolor('white')
    subplot_labels = ['a', 'b', 'c', 'd']
    
    ax_loss = axes[0]
    label_idx = 0
    
    for idx, (loss_hist, label) in enumerate(zip(loss_histories, optim_labels)):
        iterations = np.arange(1, len(loss_hist) + 1)
        marker_interval = max(1, len(loss_hist) // 8)
        
        ax_loss.semilogy(
            iterations, loss_hist,
            label=f'{label}',
            color=color_palette['optimizers'][idx],
            linestyle=line_styles[idx],
            linewidth=2.2,
            marker=markers[idx],
            markevery=marker_interval,
            markersize=6,
            alpha=0.9
        )
    
    ax_loss.set_xlabel('Iteration')
    ax_loss.set_ylabel('Loss (Log Scale)')
    ax_loss.set_title(f'({subplot_labels[label_idx]}) Optimizer Training Loss Comparison', 
                      pad=10)
    ax_loss.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
    ax_loss.set_xlim(left=0)
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)
    
    def plot_solution_with_error(ax, optim_idx: int, label_idx: int) -> None:
        """Helper for the ablation experiment."""
        label = optim_labels[optim_idx]
        color = color_palette['optimizers'][optim_idx]
        linestyle = line_styles[optim_idx]
        
        res = results[label]
        u_pred = res['u_pred'].flatten()
        pointwise_error = res['pointwise_error'].flatten()
        
        ax.plot(
            x_test_np, u_exact_np.flatten(),
            color=color_palette['exact'],
            linestyle='-',
            linewidth=2.5,
            label='Exact',
            zorder=3
        )
        ax.plot(
            x_test_np, u_pred,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            label=f'Prediction',
            alpha=0.85,
            zorder=2
        )
        
        ax.set_xlabel('Position $x$')
        ax.set_ylabel('$u(x)$')
        ax.set_title(
            f'({subplot_labels[label_idx]}) {label} (CGMPINN)',
            pad=8
        )
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
        ax.grid(True, linestyle='-', alpha=0.25)
        ax.spines['top'].set_visible(False)
        
        ax_err = ax.twinx()
        ax_err.fill_between(
            x_test_np, 0, np.abs(pointwise_error),
            color=color_palette['error'],
            alpha=0.2,
            label='|Error|'
        )
        ax_err.plot(
            x_test_np, np.abs(pointwise_error),
            color=color_palette['error'],
            linestyle=':',
            linewidth=1.5,
            alpha=0.7
        )
        ax_err.set_ylabel('Pointwise Error (Log Scale)', color=color_palette['error'])
        ax_err.tick_params(axis='y', labelcolor=color_palette['error'])
        ax_err.spines['right'].set_color(color_palette['error'])
        ax_err.spines['top'].set_visible(False)
        ax_err.set_yscale('log')

        ax_err.set_ylim(bottom=1e-10)
    
    plot_solution_with_error(axes[1], optim_idx=0, label_idx=1)
    plot_solution_with_error(axes[2], optim_idx=1, label_idx=2)
    plot_solution_with_error(axes[3], optim_idx=2, label_idx=3)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25, top=0.90)
    plt.show()

def run_1D_poisson_ablation():
    n_layers = 4
    input_dim = 1
    output_dim = 1
    hidden_dim = 50
    n_pde = 1500
    n_test = 200
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
    problem_name = "1D_poisson"
    use_relobralo_for_two_stage = False

    print("Generating training and test data...")
    pde_data, test_data = generate_data(n_pde=n_pde, n_test=n_test)
    model_temp = CGMPINN(input_dim, output_dim, hidden_dim, 1, TanhActivation())
    u_exact = model_temp.analytical_solution(test_data)
    u_exact_np = u_exact.numpy()

    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}

    for activation_name, activation_fn in activations.items():
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        print("\n" + "=" * 80)
        print(f"Starting ablation comparison - activation: {activation_name} | methods: GMM / CL / CGM")
        print("=" * 80)

        for method_name in method_labels:
            weight_strategy = method_name.lower()
            save_path = build_save_path(problem_name, activation_name, method_name)
            model = CGMPINN(
                input_dim, output_dim, hidden_dim, n_layers,
                activation=activation_fn, use_curriculum=True, gmm_kwargs=gmm_kwargs,
                use_relobralo=use_relobralo_for_two_stage, relobralo_kwargs=relobralo_kwargs,
                weight_strategy=weight_strategy
            )
            print(f"\n--- {activation_name} + {method_name} (Adam->L-BFGS) ---")

            if method_name == 'CGM' or os.path.exists(save_path):
                save_dict = torch.load(save_path, map_location='cpu', weights_only=False)
                model.load_state_dict(save_dict['model_state_dict'])
                loss_history = save_dict['loss_history']
                training_time = save_dict['training_time']
                print(f"Loaded {method_name} results from: {save_path}")
            else:
                loss_history = []
                start_time = time.time()
                train_adam_lbfgs(
                    model=model,
                    pde_data=pde_data,
                    loss_history=loss_history,
                    lr_lbfgs=lr_adam_lbfgs,
                    adam_epochs=adam_lbfgs_adam_epochs,
                    lbfgs_max_iter=adam_lbfgs_lbfgs_iter
                )
                training_time = time.time() - start_time
                save_training_artifacts(
                    save_path=save_path,
                    model=model,
                    loss_history=loss_history,
                    final_total_loss=model.final_total_loss(pde_data)[0],
                    training_time=training_time,
                    activation_name=activation_name,
                    weight_strategy=weight_strategy,
                    hyper_parameters={
                        'n_layers': n_layers,
                        'hidden_dim': hidden_dim,
                        'adam_epochs': adam_lbfgs_adam_epochs,
                        'lbfgs_max_iter': adam_lbfgs_lbfgs_iter
                    }
                )
                print(f"Saved {method_name} results to: {save_path}")

            l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model, test_data, u_exact)
            final_total, final_pde, final_boundary = model.final_total_loss(pde_data)
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

    plot_combined_results(activation_loss_histories, activation_results, test_data, u_exact_np, method_labels)

    print("\n" + "=" * 140)
    print("1D Poisson Adam->L-BFGS Ablation Comparison Table")
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


run_1D_poisson_ablation()
