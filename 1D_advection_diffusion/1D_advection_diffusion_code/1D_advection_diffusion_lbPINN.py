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

# Advection-diffusion equation parameters
BETA = 1.0       # Advection coefficient
NU = 1e-2        # Diffusion coefficient
SPACE_DOMAIN = (-1.0, 1.0)  # Spatial domain [-1, 1]
TIME_DOMAIN = (0.0, 1.0)    # Temporal domain [0, 1]

# Define activation function
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

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
        
        # Initialize adaptive weight parameters (for PDE, initial condition, and boundary condition losses)
        # Parameterized using log variance for numerical stability
        self.log_var_pde = nn.Parameter(torch.tensor(0.0))    # Log variance for PDE loss
        self.log_var_initial = nn.Parameter(torch.tensor(0.0)) # Log variance for initial condition loss
        self.log_var_boundary = nn.Parameter(torch.tensor(0.0))# Log variance for boundary condition loss

        # Regularization coefficient
        self.reg_coeff = 0.5

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)
        return self.layers(input_tensor)

    # Compute partial derivatives
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
        return u_t, u_x, u_xx

    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_t, u_x, u_xx = self.compute_gradients(x, t)
        pde_residual = u_t + BETA * u_x - NU * u_xx
        return pde_residual
    
    def initial_residual(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u_exact = -torch.sin(np.pi * x)  
        initial_residual = u_pred - u_exact
        return initial_residual
    
    def boundary_residual(self, t: torch.Tensor) -> torch.Tensor:
        x_left = -torch.ones_like(t, requires_grad=True)
        u_left = self.forward(x_left, t)
        u_x_left = torch.autograd.grad(
            outputs=u_left, inputs=x_left, grad_outputs=torch.ones_like(u_left),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        x_right = torch.ones_like(t, requires_grad=True)
        u_right = self.forward(x_right, t)
        u_x_right = torch.autograd.grad(
            outputs=u_right, inputs=x_right, grad_outputs=torch.ones_like(u_right),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        bc1_residual = u_left - u_right  
        bc2_residual = u_x_left - u_x_right  
        
        return torch.cat([bc1_residual, bc2_residual], dim=0)
    
    def adaptive_loss(self, pde_data, initial_data, boundary_data) -> torch.Tensor:
        x_pde, t_pde = pde_data
        x_initial, t_initial = initial_data
        t_boundary = boundary_data
        
        pde_res = self.pde_residual(x_pde, t_pde)
        initial_res = self.initial_residual(x_initial, t_initial)
        boundary_res = self.boundary_residual(t_boundary)
        
        raw_weight_pde = 0.5 * torch.exp(-self.log_var_pde)
        raw_weight_initial = 0.5 * torch.exp(-self.log_var_initial)
        raw_weight_boundary = 0.5 * torch.exp(-self.log_var_boundary)

        loss_pde = raw_weight_pde * torch.mean(pde_res ** 2)
        loss_initial = raw_weight_initial * torch.mean(initial_res ** 2)
        loss_boundary = raw_weight_boundary * torch.mean(boundary_res ** 2)
        
        reg_term = self.reg_coeff * (F.softplus(self.log_var_pde) + F.softplus(self.log_var_initial) + F.softplus(self.log_var_boundary))
        
        total_loss = loss_pde + loss_initial + loss_boundary + reg_term
        return total_loss, loss_pde, loss_initial, loss_boundary
    
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

# 2. generate training and testing data
def generate_data(
    n_pde: int,
    n_initial: int,
    n_boundary: int,
    n_test: int 
) -> tuple:
    # PDE internal points: x ∈ [-1,1], t ∈ [0,1]
    x_pde = (torch.rand(n_pde, 1) * 2 - 1).requires_grad_(True)
    t_pde = torch.rand(n_pde, 1).requires_grad_(True)
    pde_data = (x_pde, t_pde)
    
    # Initial condition points: t=0, x ∈ [-1,1]
    x_initial = (torch.rand(n_initial, 1) * 2 - 1).requires_grad_(True)
    t_initial = torch.zeros_like(x_initial, requires_grad=True)
    initial_data = (x_initial, t_initial)
    
    # Boundary condition points: t ∈ [0,1] (for x=-1 and x=1 boundaries)
    t_boundary = torch.rand(n_boundary, 1).requires_grad_(True)
    boundary_data = t_boundary
    
    # Test data (grid points)
    x_test = torch.linspace(SPACE_DOMAIN[0], SPACE_DOMAIN[1], n_test).reshape(-1, 1)
    t_test = torch.linspace(TIME_DOMAIN[0], TIME_DOMAIN[1], n_test).reshape(-1, 1)
    x_test_grid, t_test_grid = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    x_test_flat = x_test_grid.reshape(-1, 1)
    t_test_flat = t_test_grid.reshape(-1, 1)
    test_data = (x_test_flat, t_test_flat)
    
    return pde_data, initial_data, boundary_data, test_data

# 3. Adam training function (adapted for lbPINN)
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
        
        pde_res = model.pde_residual(x_pde, t_pde)
        initial_res = model.initial_residual(x_initial, t_initial)
        boundary_res = model.boundary_residual(t_boundary)
        
        raw_weight_pde = 0.5 * torch.exp(-model.log_var_pde)
        raw_weight_initial = 0.5 * torch.exp(-model.log_var_initial)
        raw_weight_boundary = 0.5 * torch.exp(-model.log_var_boundary)
        
        pde_loss = raw_weight_pde * torch.mean(pde_res ** 2)
        initial_loss = raw_weight_initial * torch.mean(initial_res ** 2)
        boundary_loss = raw_weight_boundary * torch.mean(boundary_res ** 2)
        reg_term = model.reg_coeff * (F.softplus(model.log_var_pde) + F.softplus(model.log_var_initial) + F.softplus(model.log_var_boundary))
        total_loss = pde_loss + initial_loss + boundary_loss + reg_term
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append({
            'total': total_loss.item(),
            'pde': pde_loss.item(),
            'initial': initial_loss.item(),
            'boundary': boundary_loss.item()
        })
        
        if epoch % 5 == 0:
            weights = model.get_adaptive_weights()
            weights['total'] = total_loss.item()  
            weight_history.append(weights) 
        
        if (epoch + 1) % 1000 == 0:
            weights = model.get_adaptive_weights()
            print(
                f'Epoch {epoch+1:4d} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss.item():.2e} | Initial Loss: {initial_loss.item():.2e} | Boundary Loss: {boundary_loss.item():.2e} '
                f'| Weights: [PDE: {weights["raw_weight_pde"]:.2e}, Initial: {weights["raw_weight_initial"]:.2e}, Boundary: {weights["raw_weight_boundary"]:.2e}]'
            )

# 4. L-BFGS training function (adapted for lbPINN)
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

    # Define closure function (required by L-BFGS)
    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, initial_loss_val, boundary_loss_val
        
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
        
        # 5. Calculate total loss
        total_loss = pde_loss + initial_loss + boundary_loss + reg_term
        
        # 6. Backpropagation
        total_loss.backward()
        
        # 7. Save current component values (for printing)
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
    print("\n=== Stage 1: Adam Global Exploration ===")
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
    
    print("\n=== Stage 2: L-BFGS Local Refinement ===")
    # L-BFGS optimizer stage
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
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        u_exact_l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()
        l2_relative_error = l2_error / u_exact_l2_norm if u_exact_l2_norm > 1e-10 else float('inf')
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        pointwise_error = torch.abs(u_pred - u_exact).numpy()
        u_pred_np = u_pred.numpy()
    
    return l2_error, l2_relative_error, linf_error, pointwise_error, u_pred_np

# 7. Visualization functions
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
        
            plt.subplot(4, 1, 2)
            plt.plot(epochs, raw_log_var_pde, label='log_var pde', linewidth=2)
            plt.plot(epochs, raw_log_var_initial, label='log_var initial', linewidth=2)
            plt.plot(epochs, raw_log_var_boundary, label='log_var boundary', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Log-Variance', fontsize=12)
            plt.title(f'Log-Variance Evolution (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.legend()
            
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

# 8. Main training routine
def train_1D_advection_diffusion_lbPINN():
    # 1. Hyperparameter settings
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
    epochs_adam = 15000    
    max_iter_lbfgs = 15000    
    adam_lbfgs_adam_epochs = 5000  
    adam_lbfgs_lbfgs_iter = 10000  
    
    # Activation functions dictionary
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
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
    
    # Compute analytical solution of the advection-diffusion equation
    pi = np.pi
    u_exact = -torch.exp(-NU * pi**2 * t_test) * torch.sin(pi * (x_test - BETA * t_test))
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}
    activation_weight_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation: {activation_name} | Advection-Diffusion Parameters: β={BETA}, ν={NU}")
        print("="*80)
        
        current_loss_histories = []
        current_weight_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Adam training
        print(f"\n--- {activation_name} + Adam (lbPINN) ---")
        model_adam = lbPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        weight_history_adam = []
        start_time = time.time()
        train_lbpinn_with_optimizer(
            model=model_adam,
            optimizer=optimizer_adam,
            epochs=epochs_adam,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam,
            weight_history=weight_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam
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
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 L-BFGS training
        print(f"\n--- {activation_name} + L-BFGS (lbPINN) ---")
        model_lbfgs = lbPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_lbfgs = []
        weight_history_lbfgs = []
        start_time = time.time()
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
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate L-BFGS
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
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Adam→L-BFGS training
        print(f"\n--- {activation_name} + Adam→L-BFGS (lbPINN) ---")
        model_adam_lbfgs = lbPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_adam_lbfgs = []
        weight_history_adam_lbfgs = []
        start_time = time.time()
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
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam→L-BFGS
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
        
        # Save Tanh+Adam→LBFGS model
        if activation_name == 'Tanh':
            save_path = "1D_advection_diffusion_tanh_adam2lbfgs_lbpinn.pth"
            complete_save_dict = {
                'model_state_dict': model_adam_lbfgs.state_dict(), 
                'loss_history': loss_history_adam_lbfgs,            
                'weight_history': weight_history_adam_lbfgs,
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
            print(f"✅ Tanh+Adam→LBFGS model saved to: {save_path}")
        
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save current activation function results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_weight_histories[activation_name] = current_weight_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # 5. Results visualization
    plot_adaptive_weights(activation_weight_histories, optim_labels)
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # 6. Print comprehensive comparison table
    print("\n" + "="*120)
    print("Advection-Diffusion Equation - lbPINN Comprehensive Comparison Table (Adaptive Loss Weights Version)")
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
    print("Best performance of each optimizer (sorted by standard L2 error)")
    print("="*100)
    for activation_name in activation_names:
        print(f"\n[{activation_name}]")
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
                f"  Rank {i}: {res['optimizer']:<20} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_advection_diffusion_lbPINN()