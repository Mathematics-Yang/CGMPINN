import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
import matplotlib.pyplot as plt
import time

# set random seeds for reproducibility
torch.manual_seed(1224)
np.random.seed(1224)

# Equation parameters
D = 0.25               # Diffusion coefficient
r = 4.0                # Reaction rate
LAMBDA = np.sqrt(r / (6 * D))  # Wave steepness parameter
c = 5 * np.sqrt(D * r / 6)     # Wave speed
x_min, x_max = -5, 5   # Spatial domain
t_min, t_max = 0, 2    # Temporal domain

# Define activation function
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# Core of LNN-PINN: Liquid Residual Block
class LiquidResidualBlock(nn.Module):
    """
    Lightweight liquid residual block (core of LNN-PINN)
    Core: Learnable gating parameters α and β, adaptively regulate information flow, maintain model compactness
    Structure: Linear -> Activation -> Gated fusion -> Residual connection
    """
    def __init__(self, hidden_dim, activation):
        super(LiquidResidualBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation  
        self.linear = nn.Linear(hidden_dim, hidden_dim)  
        
        # Learnable gating parameters (initialized close to 1 to retain original information)
        self.alpha = nn.Parameter(torch.ones(1, hidden_dim) * 0.9)  # Information retention weight
        self.beta = nn.Parameter(torch.ones(1, hidden_dim) * 0.1)   # New information weight
        self.softplus = nn.Softplus()  # Ensure gating parameters are non-negative

    def forward(self, x):
        # 1. Original input residual (retain original information)
        residual = x
        # 2. Linear transformation + activation (extract new features)
        new_features = self.linear(x)
        new_features = self.activation(new_features)
        # 3. Liquid gated fusion (adaptively regulate new/old information ratio)
        alpha = self.softplus(self.alpha)  # Ensure non-negative
        beta = self.softplus(self.beta)    # Ensure non-negative
        gated_features = alpha * residual + beta * new_features
        # 4. Residual connection (ensure training stability)
        output = self.activation(gated_features)
        return output

# LNN-PINN model
class LNN_PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation):
        super(LNN_PINN, self).__init__()
        self.activation = activation 
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_blocks = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.hidden_blocks.append(LiquidResidualBlock(hidden_dim, activation))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)
        input_tensor = self.input_layer(input_tensor)
        input_tensor = self.activation(input_tensor)
        for block in self.hidden_blocks:
            input_tensor = block(input_tensor)
        input_tensor = self.output_layer(input_tensor)
        return input_tensor

    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return torch.mean(pde_residual ** 2)
    
    def initial_loss(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u_exact = fisher_kpp_exact_solution(x, t0)  
        initial_residual = u_pred - u_exact
        return torch.mean(initial_residual ** 2)
    
    def boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        x_left = torch.full_like(t, x_min, requires_grad=True)
        u_left_pred = self.forward(x_left, t)
        u_left_exact = fisher_kpp_exact_solution(x_left, t)  
        x_right = torch.full_like(t, x_max, requires_grad=True)
        u_right_pred = self.forward(x_right, t)
        u_right_exact = fisher_kpp_exact_solution(x_right, t) 
        boundary_loss_val = torch.mean((u_left_pred - u_left_exact) ** 2) + \
                        torch.mean((u_right_pred - u_right_exact) ** 2)
        return boundary_loss_val

    def final_total_loss(self, pde_data, initial_data, boundary_data) -> tuple[float, float, float, float]:
        x_pde, t_pde = pde_data
        x_initial, t_initial = initial_data
        t_boundary = boundary_data
        
        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde, t_pde).item()
            initial_loss_val = self.initial_loss(x_initial, t_initial).item()
            boundary_loss_val = self.boundary_loss(t_boundary).item()
            total_loss_val = pde_loss_val + initial_loss_val + boundary_loss_val
        
        return total_loss_val, pde_loss_val, initial_loss_val, boundary_loss_val

# Fisher-KPP exact solution (traveling wave solution)
def fisher_kpp_exact_solution(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    z = LAMBDA * (x - c * t)
    z_clamped = torch.clamp(z, min=-50, max=50)
    exp_z = torch.exp(z_clamped)
    u = 1.0 / (1.0 + exp_z) ** 2  
    return u

# generate training and testing data
def generate_data(
    n_pde: int,
    n_initial: int,
    n_boundary: int,
    n_test: int 
) -> tuple:
    x_pde = (x_max - x_min) * torch.rand(n_pde, 1) + x_min
    t_pde = (t_max - t_min) * torch.rand(n_pde, 1) + t_min
    x_pde.requires_grad_(True)
    t_pde.requires_grad_(True)
    pde_data = (x_pde, t_pde)
    
    x_initial = (x_max - x_min) * torch.rand(n_initial, 1) + x_min
    t_initial = torch.zeros_like(x_initial)
    x_initial.requires_grad_(True)
    t_initial.requires_grad_(True)
    initial_data = (x_initial, t_initial)
    
    t_boundary = (t_max - t_min) * torch.rand(n_boundary, 1) + t_min
    t_boundary.requires_grad_(True)
    boundary_data = t_boundary
    
    x_test = torch.linspace(x_min, x_max, n_test).reshape(-1, 1)
    t_test = torch.linspace(t_min, t_max, n_test).reshape(-1, 1)
    x_test_grid, t_test_grid = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    x_test_flat = x_test_grid.reshape(-1, 1)
    t_test_flat = t_test_grid.reshape(-1, 1)
    test_data = (x_test_flat, t_test_flat)
    
    return pde_data, initial_data, boundary_data, test_data

# Adam training function
def train_with_optimizer(
    model: LNN_PINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list
) -> None:
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    model.train()

    for epoch in range(epochs):
        pde_loss = model.pde_loss(x_pde, t_pde)
        initial_loss = model.initial_loss(x_initial, t_initial)
        boundary_loss = model.boundary_loss(t_boundary)
        total_loss = pde_loss + initial_loss + boundary_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(
                f'Epoch {epoch+1:4d} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss.item():.2e} | Initial Loss: {initial_loss.item():.2e} | '
                f'Boundary Loss: {boundary_loss.item():.2e}'
            )

# L-BFGS training function
def train_with_lbfgs(
    model: LNN_PINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    lr: float,
    max_iter: int
) -> None:
    x_pde, t_pde = pde_data
    x_initial, t_initial = initial_data
    t_boundary = boundary_data
    model.train()
    pde_loss_val = initial_loss_val = boundary_loss_val = 0.0

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, initial_loss_val, boundary_loss_val
        optimizer.zero_grad()
        pde_loss = model.pde_loss(x_pde, t_pde)
        initial_loss = model.initial_loss(x_initial, t_initial)
        boundary_loss = model.boundary_loss(t_boundary)
        total_loss = pde_loss + initial_loss + boundary_loss
        total_loss.backward()
        
        pde_loss_val = pde_loss.item()
        initial_loss_val = initial_loss.item()
        boundary_loss_val = boundary_loss.item()
        return total_loss
    
    optimizer = optim.LBFGS(
        model.parameters(),
        max_iter=1,
        max_eval=10,
        line_search_fn='strong_wolfe',
        lr=lr
    )
    
    for iter_idx in range(max_iter):
        total_loss = optimizer.step(closure)
        current_loss = total_loss.item()
        loss_history.append(current_loss)
        
        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            print(
                f'Iteration {iter_idx+1}/{max_iter} | Total Loss: {current_loss:.2e} | '
                f'PDE Loss: {pde_loss_val:.2e} | Initial Loss: {initial_loss_val:.2e} | '
                f'Boundary Loss: {boundary_loss_val:.2e}'
            )

# Two-stage training (Adam → L-BFGS)
def train_adam_lbfgs(
    model: LNN_PINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    print("\n=== Stage 1: Adam Global Exploration ===")
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    train_with_optimizer(
        model=model, optimizer=optimizer_adam, epochs=adam_epochs,
        pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
        loss_history=loss_history
    )
    
    print("\n=== Stage 2: L-BFGS Local Refinement ===")
    train_with_lbfgs(
        model=model, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
        loss_history=loss_history, lr=lr_lbfgs, max_iter=lbfgs_max_iter
    )

# Evaluation function
def evaluate_model(
    model: LNN_PINN,
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

# Visualization functions
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

# Main training function for 1D Fisher-KPP using LNN-PINN
def train_1D_fisher_kpp_LNN_PINN():
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

    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (LNN-PINN)', 'L-BFGS (LNN-PINN)', 'Adam→L-BFGS (LNN-PINN)']
    
    # Generate training and testing data (original domain of Fisher-KPP)
    print("Generating training/testing data for Fisher-KPP equation...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde, n_initial=n_initial, n_boundary=n_boundary, n_test=n_test
    )
    x_test, t_test = test_data
    
    u_exact = fisher_kpp_exact_solution(x_test, t_test)
    u_exact_np = u_exact.numpy()
    
    # Initialize result storage
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation: {activation_name} | Fisher-KPP parameters: D={D}, r={r}, c={c:.2f} (LNN-PINN)")
        print("="*80)
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Pure Adam training
        print(f"\n--- {activation_name} + Adam ---")
        model_adam = LNN_PINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
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
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam (LNN-PINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 Pure L-BFGS training
        print(f"\n--- {activation_name} + L-BFGS ---")
        model_lbfgs = LNN_PINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_lbfgs = []
        start_time = time.time()
        train_with_lbfgs(
            model=model_lbfgs, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_lbfgs, lr=lr_lbfgs_config[activation_name], max_iter=max_iter_lbfgs
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['L-BFGS (LNN-PINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Adam→L-BFGS two-stage training (best model, save)
        print(f"\n--- {activation_name} + Adam→L-BFGS ---")
        model_adam_lbfgs = LNN_PINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_adam_lbfgs = []
        start_time = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs, adam_epochs=adam_lbfgs_adam_epochs,
            lr_lbfgs=lr_adam_lbfgs, lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam→L-BFGS (LNN-PINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        
        # Save best model (Tanh+Adam→L-BFGS LNN-PINN)
        save_path = "1D_fisher_kpp_tanh_adam2lbfgs_lnn_pinn.pth"
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
        print(f"✅ Fisher-KPP LNN-PINN best model saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save all results for the current activation function
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # Visualization of results
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # Print comprehensive comparison table  
    print("\n" + "="*120)
    print("Fisher-KPP equation LNN-PINN training comprehensive comparison table (Activation Function × Optimizer)")
    print("="*120)
    header = (f"{'Activation':<10} {'Optimizer':<18} {'Training Time(s)':<12} {'L2 Error':<12} "
              f"{'L2 Relative Error':<12} {'L∞ Error':<12} {'Final Total Loss':<12}")
    print(header)
    print("-"*120)
    for activation_name in activation_names:
        for idx, optim_label in enumerate(optim_labels):
            results = activation_results[activation_name][optim_label]
            train_time = activation_training_times[activation_name][idx]
            print(
                f"{activation_name:<10} "
                f"{optim_label:<18} "
                f"{train_time:<12.2f} "
                f"{results['l2_error']:<12.2e} "
                f"{results['l2_relative_error']:<12.2e} "
                f"{results['linf_error']:<12.2e} "
                f"{results['final_total_loss']:<12.2e}"
            )
    
    # Print best performances
    print("\n" + "="*100)
    print("Best performances of each optimizer (sorted by standard L2 error - LNN-PINN)")
    print("="*100)
    for activation_name in activation_names:
        print(f"\n【{activation_name}】")
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
                f"  Rank {i}: {res['optimizer']:<18} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_fisher_kpp_LNN_PINN()