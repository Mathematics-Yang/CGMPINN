import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm  
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# set random seeds for reproducibility
torch.manual_seed(1224)
np.random.seed(1224)

# define activation functions
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# define problem parameters
ALPHA1 = 1.0    
ALPHA2 = 1.0    
GAMMA = 0.1     
BETA = GAMMA    
c = np.sqrt(GAMMA**2 + (ALPHA2 * np.pi)**2) / (ALPHA1 * np.pi)  

# 1. lbPINN model
class lbPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation):
        super(lbPINN, self).__init__()
        self.activation = activation  
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        
        # Initialize adaptive weight parameters (for PDE, initial condition, boundary condition losses)
        # Parameterized by log variance for numerical stability
        self.log_var_pde = nn.Parameter(torch.tensor(0.0))    # Log variance for PDE loss
        self.log_var_initial = nn.Parameter(torch.tensor(0.0)) # Log variance for initial condition loss
        self.log_var_boundary = nn.Parameter(torch.tensor(0.0))# Log variance for boundary condition loss
        # Regularization coefficient
        self.reg_coeff = 0.5

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1) 
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor

    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        return u_t, u_tt, u_xx

    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_t, u_tt, u_xx = self.compute_gradients(x, t)
        pde_residual = u_tt + 2 * BETA * u_t - (c ** 2) * u_xx
        return pde_residual
    
    def initial_residual(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u0_exact = torch.sin(ALPHA1 * np.pi * x)
        res1 = u_pred - u0_exact
        
        u_t, _, _ = self.compute_gradients(x, t0)
        v0_exact = -GAMMA * torch.sin(ALPHA1 * np.pi * x)
        res2 = u_t - v0_exact
        
        return torch.cat([res1, res2], dim=0)
    
    def boundary_residual(self, t: torch.Tensor) -> torch.Tensor:
        x0 = torch.zeros_like(t)
        u0_pred = self.forward(x0, t)
        u0_exact = torch.exp(-GAMMA * t) * torch.sin(ALPHA1 * np.pi * x0) * torch.cos(ALPHA2 * np.pi * t)
        
        x1 = torch.ones_like(t)
        u1_pred = self.forward(x1, t)
        u1_exact = torch.exp(-GAMMA * t) * torch.sin(ALPHA1 * np.pi * x1) * torch.cos(ALPHA2 * np.pi * t)
        
        boundary_residual1 = u0_pred - u0_exact
        boundary_residual2 = u1_pred - u1_exact
        return torch.cat([boundary_residual1, boundary_residual2], dim=0)
    
    # Adaptive loss function (based on Gaussian likelihood estimation)
    def adaptive_loss(self, pde_data, initial_data, boundary_data) -> torch.Tensor:
        """
        Adaptive loss function: based on Gaussian probabilistic model and maximum likelihood estimation
        Loss = weighted sum of residuals + regularization term (to prevent extreme weights)
        """
        x_pde, t_pde = pde_data
        x_initial, t_initial = initial_data
        t_boundary = boundary_data
        
        # Calculate residuals
        pde_res = self.pde_residual(x_pde, t_pde)
        initial_res = self.initial_residual(x_initial, t_initial)
        boundary_res = self.boundary_residual(t_boundary)
        
        # Calculate raw adaptive weights (1/(2σ²) = 0.5*exp(-log_var))
        raw_weight_pde = 0.5 * torch.exp(-self.log_var_pde)
        raw_weight_initial = 0.5 * torch.exp(-self.log_var_initial)
        raw_weight_boundary = 0.5 * torch.exp(-self.log_var_boundary)

        # Weighted residual losses
        loss_pde = raw_weight_pde * torch.mean(pde_res ** 2)
        loss_initial = raw_weight_initial * torch.mean(initial_res ** 2)
        loss_boundary = raw_weight_boundary * torch.mean(boundary_res ** 2)
        
        # Regularization term to prevent extreme weights
        reg_term = self.reg_coeff * (F.softplus(self.log_var_pde) + F.softplus(self.log_var_initial) + F.softplus(self.log_var_boundary))
        
        # Total loss
        total_loss = loss_pde + loss_initial + loss_boundary + reg_term
        return total_loss, loss_pde, loss_initial, loss_boundary
    
    # Calculate final total loss (for evaluation)
    def final_total_loss(self, pde_data, initial_data, boundary_data) -> tuple[float, float, float, float]:
        x_pde, t_pde = pde_data
        x_initial, t_initial = initial_data
        t_boundary = boundary_data
        
        with torch.enable_grad():
            total_loss, pde_loss, initial_loss, boundary_loss = self.adaptive_loss(pde_data, initial_data, boundary_data)
            return (total_loss.item(), pde_loss.item(), initial_loss.item(), boundary_loss.item())
    
    def get_adaptive_weights(self) -> dict:
        raw_weight_pde = 0.5 * torch.exp(-self.log_var_pde).item()
        raw_weight_initial = 0.5 * torch.exp(-self.log_var_initial).item()
        raw_weight_boundary = 0.5 * torch.exp(-self.log_var_boundary).item()
        raw_log_var_pde = self.log_var_pde.item()
        raw_log_var_initial = self.log_var_initial.item()
        raw_log_var_boundary = self.log_var_boundary.item()

        return {
            'raw_weight_pde': raw_weight_pde,
            'raw_weight_initial': raw_weight_initial,
            'raw_weight_boundary': raw_weight_boundary,
            'raw_log_var_pde': raw_log_var_pde,
            'raw_log_var_initial': raw_log_var_initial,
            'raw_log_var_boundary': raw_log_var_boundary,
            'raw_weight_sum': raw_weight_pde + raw_weight_initial + raw_weight_boundary,
            'total': 0.0
        }

# 2. Generate training and testing data
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

# 3. Adam/SGD training function
def train_lbpinn_with_optimizer(
    model: lbPINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    weight_history: list
) -> None:
    model.train()
    for epoch in range(epochs):
        x_pde, t_pde = pde_data
        x_initial, t_initial = initial_data
        t_boundary = boundary_data
        
        # Calculate residuals
        pde_res = model.pde_residual(x_pde, t_pde)
        initial_res = model.initial_residual(x_initial, t_initial)
        boundary_res = model.boundary_residual(t_boundary)
        
        # Calculate raw adaptive weights
        raw_weight_pde = 0.5 * torch.exp(-model.log_var_pde)
        raw_weight_initial = 0.5 * torch.exp(-model.log_var_initial)
        raw_weight_boundary = 0.5 * torch.exp(-model.log_var_boundary)
        
        # Calculate losses
        pde_loss = raw_weight_pde * torch.mean(pde_res ** 2)
        initial_loss = raw_weight_initial * torch.mean(initial_res ** 2)
        boundary_loss = raw_weight_boundary * torch.mean(boundary_res ** 2)
        reg_term = model.reg_coeff * (F.softplus(model.log_var_pde) + F.softplus(model.log_var_initial) + F.softplus(model.log_var_boundary))
        total_loss = pde_loss + initial_loss + boundary_loss + reg_term
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Record losses
        loss_history.append({
            'total': total_loss.item(),
            'pde': pde_loss.item(),
            'initial': initial_loss.item(),
            'boundary': boundary_loss.item()
        })
        
        # Record weights
        if epoch % 5 == 0:
            weights = model.get_adaptive_weights()
            weights['total'] = total_loss.item()  
            weight_history.append(weights)
        
        # Print information
        if (epoch + 1) % 1000 == 0:
            weights = model.get_adaptive_weights()
            print(
                f'Epoch {epoch+1:4d} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss.item():.2e} | Initial Loss: {initial_loss.item():.2e} | Boundary Loss: {boundary_loss.item():.2e} '
                f'| Weights: [PDE: {weights["raw_weight_pde"]:.2e}, Initial: {weights["raw_weight_initial"]:.2e}, Boundary: {weights["raw_weight_boundary"]:.2e}]'
            )

# 4. Train lbPINN model using L-BFGS optimizer
def train_lbfgs_lbpinn(
    model: lbPINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    weight_history: list,  
    lbfgs_max_iter: int,   
    lr_lbfgs: float      
) -> None:
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    
    model.train() 
    
    pde_loss_val = initial_loss_val = boundary_loss_val = 0.0
    norm_weight_pde_val = norm_weight_initial_val = norm_weight_boundary_val = 0.0

    # Define closure function (required by L-BFGS)
    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, initial_loss_val, boundary_loss_val
        nonlocal norm_weight_pde_val, norm_weight_initial_val, norm_weight_boundary_val
        
        optimizer.zero_grad()  
        
        # 1. Calculate residuals
        pde_res = model.pde_residual(x_pde, t_pde)
        initial_res = model.initial_residual(x_initial, t_initial)
        boundary_res = model.boundary_residual(t_boundary)
        
        # 2. Calculate raw adaptive weights
        raw_weight_pde = 0.5 * torch.exp(-model.log_var_pde)
        raw_weight_initial = 0.5 * torch.exp(-model.log_var_initial)
        raw_weight_boundary = 0.5 * torch.exp(-model.log_var_boundary)
        
        # 3. Calculate component losses
        pde_loss = raw_weight_pde * torch.mean(pde_res ** 2)
        initial_loss = raw_weight_initial * torch.mean(initial_res ** 2)
        boundary_loss = raw_weight_boundary * torch.mean(boundary_res ** 2)
        
        # 4. Regularization term
        reg_term = model.reg_coeff * (
            F.softplus(model.log_var_pde) + F.softplus(model.log_var_initial) + F.softplus(model.log_var_boundary)
        )
        
        # 5. Total loss
        total_loss = pde_loss + initial_loss + boundary_loss + reg_term
        
        # 6. Backpropagation
        total_loss.backward()
        
        # 7. Save current component loss values (for printing)
        pde_loss_val = pde_loss.item()
        initial_loss_val = initial_loss.item()
        boundary_loss_val = boundary_loss.item()
        
        return total_loss
    
    # Initialize L-BFGS optimizer
    optimizer = optim.LBFGS(
        model.parameters(),
        max_iter=1,  
        max_eval=10,  
        line_search_fn='strong_wolfe',  
        lr=lr_lbfgs 
    )
    
    for iter_idx in range(lbfgs_max_iter):  
        total_loss = optimizer.step(closure)
        current_loss = total_loss.item()
        
        loss_history.append({
            'total': current_loss,
            'pde': pde_loss_val,
            'initial': initial_loss_val,
            'boundary': boundary_loss_val
        })
        
        if iter_idx % 5 == 0:
            weights = model.get_adaptive_weights()
            weights['total'] = current_loss 
            weight_history.append(weights)
        
        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            print(
                f'Iteration {iter_idx+1}/{lbfgs_max_iter} | Total Loss: {current_loss:.2e} | '
                f'PDE Loss: {pde_loss_val:.2e} | Initial Condition Loss: {initial_loss_val:.2e} | '
                f'Boundary Condition Loss: {boundary_loss_val:.2e} | '
                f'Weights: [PDE: {weights["raw_weight_pde"]:.2e}, Initial: {weights["raw_weight_initial"]:.2e}, Boundary: {weights["raw_weight_boundary"]:.2e}]'
            )

# 5. Two-stage training function: Adam + L-BFGS (lbPINN version)
def train_adam_lbfgs_lbpinn(
    model: lbPINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    weight_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    """Two-stage training: first Adam global exploration, then L-BFGS local fine-tuning (lbPINN version)"""
    print("=== Training with Adam optimizer ===")
    # Adam optimizer stage
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    train_lbpinn_with_optimizer(
        model=model,
        optimizer=optimizer_adam,
        epochs=adam_epochs,
        pde_data=pde_data,
        initial_data=initial_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        weight_history=weight_history
    )
    # L-BFGS optimizer stage
    print("\n=== Fine-tuning with L-BFGS optimizer ===")
    train_lbfgs_lbpinn(
        model=model,
        pde_data=pde_data,
        initial_data=initial_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        weight_history=weight_history,
        lbfgs_max_iter=lbfgs_max_iter,
        lr_lbfgs=lr_lbfgs
    )

# 6. evaluation function
def evaluate_model(
    model: lbPINN,
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

# 7. visualization function
def plot_adaptive_weights(weight_histories: dict, optim_labels: list) -> None:
    for activation_name, optim_weights_list in weight_histories.items():
        for opt_idx, (weights_list, opt_label) in enumerate(zip(optim_weights_list, optim_labels)):
            if not weights_list:  
                continue
                
            plt.figure(figsize=(10, 12))
            epochs = [i*5 for i in range(len(weights_list))]
            
            raw_weight_pde = [w['raw_weight_pde'] for w in weights_list]
            raw_weight_initial = [w['raw_weight_initial'] for w in weights_list]
            raw_weight_boundary = [w['raw_weight_boundary'] for w in weights_list]
            raw_log_var_pde = [w['raw_log_var_pde'] for w in weights_list]
            raw_log_var_initial = [w['raw_log_var_initial'] for w in weights_list]
            raw_log_var_boundary = [w['raw_log_var_boundary'] for w in weights_list]
            
            # 1. Raw weights
            plt.subplot(4, 1, 1)
            plt.plot(epochs, raw_weight_pde, label='raw pde weight', linewidth=2)
            plt.plot(epochs, raw_weight_initial, label='raw initial weight', linewidth=2)
            plt.plot(epochs, raw_weight_boundary, label='raw boundary weight', linewidth=2)
            plt.yscale('log')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Raw Weight (Log Scale)', fontsize=12)
            plt.title(f'Raw Adaptive Weights (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
            # 2. Log-Variance
            plt.subplot(4, 1, 2)
            plt.plot(epochs, raw_log_var_pde, label='log_var pde', linewidth=2)
            plt.plot(epochs, raw_log_var_initial, label='log_var initial', linewidth=2)
            plt.plot(epochs, raw_log_var_boundary, label='log_var boundary', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Log-Variance', fontsize=12)
            plt.title(f'Log-Variance Evolution (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.legend()
            
            # 3. Total Loss
            plt.subplot(4, 1, 3)
            loss_history = [w['total'] for w in weights_list]
            plt.plot(epochs, loss_history, label='Total Loss', linewidth=2, color='black')
            plt.yscale('log')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (Log Scale)', fontsize=12)
            plt.title(f'Loss Evolution (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # 4. Normalized weight proportions
            plt.subplot(4, 1, 4)
            weight_sum = [w['raw_weight_sum'] for w in weights_list]
            norm_pde = [p / s if s > 0 else 0 for p, s in zip(raw_weight_pde, weight_sum)]
            norm_initial = [p / s if s > 0 else 0 for p, s in zip(raw_weight_initial, weight_sum)]
            norm_boundary = [p / s if s > 0 else 0 for p, s in zip(raw_weight_boundary, weight_sum)]
            plt.plot(epochs, norm_pde, label='PDE proportion', linewidth=2)
            plt.plot(epochs, norm_initial, label='Initial proportion', linewidth=2)
            plt.plot(epochs, norm_boundary, label='Boundary proportion', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Weight Proportion', fontsize=12)
            plt.title(f'Normalized Weight Proportions (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.ylim(-0.05, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

# 8. visualization function
def plot_loss_curves_by_activation(activation_loss_histories: dict, optim_labels: list) -> None:
    for activation_name, loss_histories in activation_loss_histories.items():
        plt.figure(figsize=(10, 6))
        for loss_history, label in zip(loss_histories, optim_labels):
            if not loss_history: 
                continue
                
            total_losses = [item['total'] for item in loss_history]
            plt.plot(total_losses, label=label, linewidth=2)
        
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
        
        # First column: spatial profiles at different time points
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
        
        # Second column: temporal evolution at x=0.5
        ax2 = axes[row_idx, 1]
        x_mid_idx = np.argmin(np.abs(x_coords - 0.5))
        t_vals = t_coords 
        
        ax2.plot(t_vals, u_exact_grid[x_mid_idx, :], 
                 'b-', linewidth=2.5, label='Analytical Solution')
        ax2.plot(t_vals, u_pred_grid[x_mid_idx, :], 
                 'r--', linewidth=2, label=f'{optim_label} Prediction')
        
        # Plot damping envelope
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
        
        # Third column: error analysis at x=0.5
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

# 9. main function
def train_1D_wave_lbPINN_with_activations():
    # 1. hyperparameter settings
    n_layers = 4
    input_dim = 2
    output_dim = 1
    hidden_dim = 50
    n_pde = 2000
    n_initial = 300
    n_boundary = 300
    n_test = 100
    lr_lbfgs_config = {'Tanh': 1.0}
    lr_adam_lbfgs = 0.8
    epochs_sgd_adam = 15000
    max_iter_lbfgs = 15000
    adam_lbfgs_adam_epochs = 5000
    adam_lbfgs_lbfgs_iter = 10000
    
    # Activation functions dictionary
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    
    # Optimizer labels
    optim_labels = ['Adam (lbPINN)', 'L-BFGS (lbPINN)', 'Adam→L-BFGS (lbPINN)']
    
    # 2. Generate data
    print("Generating training and testing data...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde,
        n_initial=n_initial,
        n_boundary=n_boundary,
        n_test=n_test
    )
    x_test, t_test = test_data
    # Analytical solution of the wave equation: u(x,t) = e^(-γt) sin(α1πx) cos(α2πt)
    u_exact = torch.exp(-GAMMA * t_test) * torch.sin(ALPHA1 * np.pi * x_test) * torch.cos(ALPHA2 * np.pi * t_test)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables (adapted for lbPINN loss and weight records)
    activation_loss_histories = {}  # {activation_name: [Adam loss, L-BFGS loss, two-stage loss]}
    activation_weight_histories = {} # {activation_name: [Adam weights, L-BFGS weights, two-stage weights]}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation Function: {activation_name} | Damped 1D Wave Equation")
        print("="*80)
        
        current_loss_histories = []
        current_weight_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Train Adam (lbPINN)
        print(f"\n--- {activation_name} + Adam (lbPINN) ---")
        model_adam = lbPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        weight_history_adam = []
        start_time_adam = time.time()
        train_lbpinn_with_optimizer(
            model=model_adam,
            optimizer=optimizer_adam,
            epochs=epochs_sgd_adam,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam,
            weight_history=weight_history_adam
        )
        training_time_adam = time.time() - start_time_adam
        current_training_times.append(training_time_adam)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam (lbPINN)'] = {
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
        current_weight_histories.append(weight_history_adam)
        print(f"Adam training time: {training_time_adam:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 Train L-BFGS (lbPINN)
        print(f"\n--- {activation_name} + L-BFGS (lbPINN) ---")
        model_lbfgs = lbPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_lbfgs = []
        weight_history_lbfgs = []
        start_time_lbfgs = time.time()
        train_lbfgs_lbpinn(
            model=model_lbfgs,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_lbfgs,
            weight_history=weight_history_lbfgs,
            lbfgs_max_iter=max_iter_lbfgs,
            lr_lbfgs=lr_lbfgs_config[activation_name]
        )
        training_time_lbfgs = time.time() - start_time_lbfgs
        current_training_times.append(training_time_lbfgs)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['L-BFGS (lbPINN)'] = {
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
        current_weight_histories.append(weight_history_lbfgs)
        print(f"L-BFGS training time: {training_time_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Train Adam→L-BFGS (lbPINN)
        print(f"\n--- {activation_name} + Adam→L-BFGS (lbPINN) ---")
        model_adam_lbfgs = lbPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_adam_lbfgs = []
        weight_history_adam_lbfgs = []
        start_time_adam_lbfgs = time.time()
        train_adam_lbfgs_lbpinn(
            model=model_adam_lbfgs,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs,
            weight_history=weight_history_adam_lbfgs,
            lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        training_time_adam_lbfgs = time.time() - start_time_adam_lbfgs
        current_training_times.append(training_time_adam_lbfgs)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam→L-BFGS (lbPINN)'] = {
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
        current_weight_histories.append(weight_history_adam_lbfgs)
        
        # Save model
        if activation_name == 'Tanh':  
            # Define save path
            save_path = "1D_wave_tanh_adam2lbfgs_lbpinn.pth"
    
            # Package all contents to save
            complete_save_dict = {
                'model_state_dict': model_adam_lbfgs.state_dict(), 
                'loss_history': loss_history_adam_lbfgs,  
                'weight_history': weight_history_adam_lbfgs,          
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
        
        # Save results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_weight_histories[activation_name] = current_weight_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # 5. Visualization
    # 5.1 Plot adaptive weight curves
    plot_adaptive_weights(activation_weight_histories, optim_labels)
    
    # 5.2 Plot loss curves
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    
    # 5.3 Plot solution comparison
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)

    plot_damping_effect(x_test, t_test,
                        {opt_label: activation_results['Tanh'][opt_label]['u_pred'] for opt_label in optim_labels},
                        u_exact.numpy())
    
    # 6. Print comprehensive comparison table
    print("\n" + "="*120)
    print("lbPINN All Activation Functions × Optimizers Comprehensive Comparison Table (1D Wave Equation with Damping)")
    print("="*120)
    header = (f"{'Activation':<10} {'Optimizer':<20} {'Training Time(s)':<12} {'Standard L2 Error':<12} "
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
    
    # 7. Print best performance of each activation function
    print("\n" + "="*100)
    print("lbPINN Best Performance of Each Activation Function (Sorted by Standard L2 Error)")
    print("="*100)
    for activation_name in activation_names:
        print(f"\n【{activation_name}】")
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
        # Print sorted results
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<20} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_wave_lbPINN_with_activations()