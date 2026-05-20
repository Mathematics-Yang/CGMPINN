import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm  
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

# set random seed for reproducibility
torch.manual_seed(1224)
np.random.seed(1224)

# define activation function
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# define equation parameters
ALPHA1 = 1.0  # spatial parameter (controls spatial oscillation frequency)
ALPHA2 = 2.0  # temporal parameter (controls temporal oscillation frequency)
k = 10.0 # spatial parameter (controls steepness of hyperbolic tangent function)

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
        
        # initialize adaptive weight parameters (for PDE, initial condition, boundary condition losses)
        # parameterize with log variance for numerical stability
        self.log_var_pde = nn.Parameter(torch.tensor(0.0))    # log variance of PDE loss
        self.log_var_initial = nn.Parameter(torch.tensor(0.0)) # log variance of initial condition loss
        self.log_var_boundary = nn.Parameter(torch.tensor(0.0))# log variance of boundary condition loss

        # regularization coefficient
        self.reg_coeff = 0.5

    # forward propagation
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)  
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor

    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.forward(x, t)  
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones_like(u),
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
        return u_t, u_xx

    # define source term f(x,t)
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

    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_t, u_xx = self.compute_gradients(x, t)
        f = self.source_term(x, t)
        pde_residual = u_t - u_xx - f 
        return pde_residual
    
    def initial_residual(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u_exact = (torch.sin(ALPHA1 * np.pi * x) + torch.tanh(k * x)) * torch.sin(ALPHA2 * np.pi * t0)
        initial_residual = u_pred - u_exact
        return initial_residual
    
    def boundary_residual(self, t: torch.Tensor) -> torch.Tensor:
        x0 = torch.zeros_like(t)
        u0_pred = self.forward(x0, t)
        u0_exact = (torch.sin(ALPHA1 * np.pi * x0) + torch.tanh(k * x0)) * torch.sin(ALPHA2 * np.pi * t)
        x1 = torch.ones_like(t)
        u1_pred = self.forward(x1, t)
        u1_exact = (torch.sin(ALPHA1 * np.pi * x1) + torch.tanh(k * x1)) * torch.sin(ALPHA2 * np.pi * t)
        boundary_residual1 = u0_pred - u0_exact
        boundary_residual2 = u1_pred - u1_exact

        return torch.cat([boundary_residual1, boundary_residual2], dim=0)
    
    # adaptive loss function (based on Gaussian likelihood estimation)
    def adaptive_loss(self, pde_data, initial_data, boundary_data) -> torch.Tensor:
        """
        Adaptive loss function: based on Gaussian probabilistic model and maximum likelihood estimation
        Loss = weighted sum of residual terms + regularization term (to prevent extreme weights)
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
    
    # Get current adaptive weights (for monitoring)
    def get_adaptive_weights(self) -> dict:
        # Raw weights and log_var
        raw_weight_pde = 0.5 * torch.exp(-self.log_var_pde).item()
        raw_weight_initial = 0.5 * torch.exp(-self.log_var_initial).item()
        raw_weight_boundary = 0.5 * torch.exp(-self.log_var_boundary).item()
        raw_log_var_pde = self.log_var_pde.item()
        raw_log_var_initial = self.log_var_initial.item()
        raw_log_var_boundary = self.log_var_boundary.item()

        return {
            # Raw weights/Log_var
            'raw_weight_pde': raw_weight_pde,
            'raw_weight_initial': raw_weight_initial,
            'raw_weight_boundary': raw_weight_boundary,
            'raw_log_var_pde': raw_log_var_pde,
            'raw_log_var_initial': raw_log_var_initial,
            'raw_log_var_boundary': raw_log_var_boundary,
            'raw_weight_sum': raw_weight_pde + raw_weight_initial + raw_weight_boundary,
            # Total loss (default value, updated during training)
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
            weights['total'] = total_loss.item()  # Add total loss field
            weight_history.append(weights) # Record current weights
        
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
    
    # Variables to store the latest component loss values
    pde_loss_val = initial_loss_val = boundary_loss_val = 0.0
    norm_weight_pde_val = norm_weight_initial_val = norm_weight_boundary_val = 0.0

    # Define closure function
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
        line_search_fn='strong_wolfe',  # Line search strategy
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
        
        # Print iteration information
        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            print(
                f'Iteration {iter_idx+1}/{lbfgs_max_iter} | Total Loss: {current_loss:.2e} | '
                f'PDE Loss: {pde_loss_val:.2e} | Initial Condition Loss: {initial_loss_val:.2e} | '
                f'Boundary Condition Loss: {boundary_loss_val:.2e} | '
                f'Weights: [PDE: {weights["raw_weight_pde"]:.2e}, Initial: {weights["raw_weight_initial"]:.2e}, Boundary: {weights["raw_weight_boundary"]:.2e}]'
            )

# 5. Two-stage training function (Adam→L-BFGS)
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
    print("=== Using Adam optimizer for training ===")
    # Adam optimizer phase
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
    # L-BFGS optimizer phase
    print("\n=== Using L-BFGS optimizer for fine-tuning ===")
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

# 6. Evaluation function
def evaluate_model(
    model: lbPINN,
    test_data: tuple,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    x_test, t_test = test_data
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test, t_test)
        
        # 1. L2 error
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        
        # 2. L2 relative error
        u_exact_l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()
        l2_relative_error = l2_error / u_exact_l2_norm if u_exact_l2_norm > 1e-10 else float('inf')
        
        # 3. L∞ error
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        
        # 4. Pointwise error
        pointwise_error = torch.abs(u_pred - u_exact).numpy()
        
        # 5. Predicted values (numpy format)
        u_pred_np = u_pred.numpy()
    
    return l2_error, l2_relative_error, linf_error, pointwise_error, u_pred_np

# 7. Visualization function
def plot_adaptive_weights(weight_histories: dict, optim_labels: list) -> None:
    for activation_name, optim_weights_list in weight_histories.items():
        for opt_idx, (weights_list, opt_label) in enumerate(zip(optim_weights_list, optim_labels)):
            if not weights_list:  
                continue
                
            plt.figure(figsize=(10, 12))
            epochs = [i*5 for i in range(len(weights_list))]
            
            # Raw weights/Log_var
            raw_weight_pde = [w['raw_weight_pde'] for w in weights_list]
            raw_weight_initial = [w['raw_weight_initial'] for w in weights_list]
            raw_weight_boundary = [w['raw_weight_boundary'] for w in weights_list]
            raw_log_var_pde = [w['raw_log_var_pde'] for w in weights_list]
            raw_log_var_initial = [w['raw_log_var_initial'] for w in weights_list]
            raw_log_var_boundary = [w['raw_log_var_boundary'] for w in weights_list]
            
            # 1. raw weights
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
            
            # 3. Total loss
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

# 8. Visualization function
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
            
            # Reshape to grid form
            u_pred_grid = u_pred.reshape(n_test, n_test)
            u_exact_grid = u_exact_np.reshape(n_test, n_test)
            error_grid = np.abs(u_pred_grid - u_exact_grid)
            x_grid = x_test.reshape(n_test, n_test)
            t_grid = t_test.reshape(n_test, n_test)
            
            # 1st column: Predicted solution
            ax1 = axes[idx, 0]
            im1 = ax1.pcolormesh(t_grid, x_grid, u_pred_grid, cmap=cm.jet, shading='gouraud')
            ax1.set_xlabel('Time (t)', fontsize=10)
            ax1.set_ylabel('Position (x)', fontsize=10)
            ax1.set_title(f'{optim_label} - Predicted', fontsize=12)
            plt.colorbar(im1, ax=ax1, shrink=0.6, aspect=5)
            
            # 2nd column: Analytical solution
            ax2 = axes[idx, 1]
            im2 = ax2.pcolormesh(t_grid, x_grid, u_exact_grid, cmap=cm.jet, shading='gouraud')
            ax2.set_title('Analytical Solution', fontsize=12)
            plt.colorbar(im2, ax=ax2, shrink=0.6, aspect=5)
            ax2.set_xlabel('Time (t)', fontsize=10)
            ax2.set_ylabel('Position (x)', fontsize=10)
            
            # 3rd column: Absolute error
            ax3 = axes[idx, 2]
            im3 = ax3.pcolormesh(t_grid, x_grid, error_grid, cmap=cm.viridis, shading='gouraud')
            ax3.set_xlabel('Time (t)', fontsize=10)
            ax3.set_ylabel('Position (x)', fontsize=10)
            ax3.set_title(f'{optim_label} - Absolute Error', fontsize=12)
            plt.colorbar(im3, ax=ax3, shrink=0.6, aspect=5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

# 9. Main training function
def train_1D_heat_lbPINN_with_activations_and_source_term():
    # 1. Hyperparameter settings
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
    u_exact = (torch.sin(ALPHA1 * np.pi * x_test) + torch.tanh(k * x_test)) * torch.sin(ALPHA2 * np.pi * t_test)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}  
    activation_weight_histories = {} 
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name}")
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
        if activation_name == 'Tanh': 
            # Define save path
            save_path = "1D_heat_tanh_adam2lbfgs_lbpinn.pth"
    
            # Pack all contents to be saved
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
            print(f"✅ Tanh+Adam→L-BFGS complete information saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {training_time_adam_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save results for current activation function
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
    
    # 6. Print comprehensive comparison table
    print("\n" + "="*120)
    print("lbPINN All Activation Functions × Optimizers Comprehensive Comparison Table")
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
    
    # 7. print best results by activation function
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
train_1D_heat_lbPINN_with_activations_and_source_term()