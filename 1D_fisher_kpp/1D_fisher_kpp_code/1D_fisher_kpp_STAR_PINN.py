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

# Fisher-KPP parameters
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

# Fisher-KPP exact traveling wave solution (Ablowitz-Zeppetella)
def fisher_kpp_exact_solution(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    z = LAMBDA * (x - c * t)
    z_clamped = torch.clamp(z, min=-50, max=50)  # Numerical stability
    exp_z = torch.exp(z_clamped)
    u = 1.0 / (1.0 + exp_z) ** 2  
    return u

# Lightweight PINN block (basic module of STAR-PINN)
class LightweightPINNBlock(nn.Module):
    """Lightweight PINN block for stacked adaptive residual structure in STAR-PINN"""
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation):
        super(LightweightPINNBlock, self).__init__()
        self.activation = activation
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# STAR-PINN model
class STAR_PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers_per_block, activation, n_blocks=3):
        super(STAR_PINN, self).__init__()
        self.n_blocks = n_blocks  # Number of stacked PINN blocks
        self.activation = activation
        
        # Create stacked lightweight PINN blocks
        self.pinn_blocks = nn.ModuleList([
            LightweightPINNBlock(input_dim, output_dim, hidden_dim, n_layers_per_block, activation)
            for _ in range(n_blocks)
        ])
        
        # Adaptive residual weights (trainable, initialized to 0.5)
        self.adaptive_weights = nn.Parameter(torch.ones(n_blocks - 1) * 0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)
        current_output = self.pinn_blocks[0](input_tensor)
        # Subsequent blocks: adaptive residual fusion (sigmoid restricts weights to [0,1])
        for i in range(1, self.n_blocks):
            block_output = self.pinn_blocks[i](input_tensor)
            alpha = torch.sigmoid(self.adaptive_weights[i-1])  # Adaptive weight
            current_output = block_output + alpha * current_output
        return current_output

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

# Data generation
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

# Training function (adapted for STAR-PINN)
def train_with_optimizer(
    model: STAR_PINN,
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
                f'PDE Loss: {pde_loss.item():.2e} | Initial Condition Loss: {initial_loss.item():.2e} | '
                f'Boundary Condition Loss: {boundary_loss.item():.2e}'
            )

def train_with_lbfgs(
    model: STAR_PINN,
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
                f'PDE Loss: {pde_loss_val:.2e} | Initial Condition Loss: {initial_loss_val:.2e} | '
                f'Boundary Condition Loss: {boundary_loss_val:.2e}'
            )

def train_adam_lbfgs(
    model: STAR_PINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    print("\n=== Phase 1: Adam Global Exploration ===")
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    train_with_optimizer(
        model=model, optimizer=optimizer_adam, epochs=adam_epochs,
        pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
        loss_history=loss_history
    )
    
    print("\n=== Phase 2: L-BFGS Local Refinement ===")
    train_with_lbfgs(
        model=model, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
        loss_history=loss_history, lr=lr_lbfgs, max_iter=lbfgs_max_iter
    )

# Evaluation function (adapted for STAR-PINN) 
def evaluate_model(
    model: STAR_PINN,
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

# Main function (adapted for STAR-PINN hyperparameters) 
def train_1D_fisher_kpp_STAR_PINN():
    n_layers_per_block = 2    
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

    # Activation functions and optimizer labels (adapted for STAR-PINN)
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (STAR-PINN)', 'L-BFGS (STAR-PINN)', 'Adam→L-BFGS (STAR-PINN)']
    
    # Generate data
    print("Generating training/testing data for the Fisher-KPP equation...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde, n_initial=n_initial, n_boundary=n_boundary, n_test=n_test
    )
    x_test, t_test = test_data
    u_exact = fisher_kpp_exact_solution(x_test, t_test)
    u_exact_np = u_exact.numpy()
    
    # Results storage
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation: {activation_name} | Fisher-KPP parameters: D={D}, r={r}, c={c:.2f}")
        print(f"STAR-PINN configuration: {3} stacked blocks × {n_layers_per_block} hidden layers per block")
        print("="*80)
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Pure Adam training
        print(f"\n--- {activation_name} + Adam ---")
        model_adam = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, activation_fn)
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
        current_results['Adam (STAR-PINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 Pure L-BFGS training
        print(f"\n--- {activation_name} + L-BFGS ---")
        model_lbfgs = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, activation_fn)
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
        current_results['L-BFGS (STAR-PINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Adam→L-BFGS two-stage training
        print(f"\n--- {activation_name} + Adam→L-BFGS ---")
        model_adam_lbfgs = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, activation_fn)
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
        current_results['Adam→L-BFGS (STAR-PINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        
        # Save the best STAR-PINN model
        save_path = "1D_fisher_kpp_tanh_adam2lbfgs_star_pinn.pth"
        complete_save_dict = {
            'model_state_dict': model_adam_lbfgs.state_dict(),
            'loss_history': loss_history_adam_lbfgs,
            'final_total_loss': final_total,
            'training_time': train_time,
            'activation_name': activation_name,
            'hyper_parameters': {
                'n_layers_per_block': n_layers_per_block, 
                'hidden_dim': hidden_dim,
                'adam_epochs': adam_lbfgs_adam_epochs, 
                'lbfgs_max_iter': adam_lbfgs_lbfgs_iter
            }
        }
        torch.save(complete_save_dict, save_path)
        print(f"✅ Fisher-KPP STAR-PINN best model saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # Visualization of results
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # Print comprehensive comparison table
    print("\n" + "="*120)
    print("Fisher-KPP equation STAR-PINN training comprehensive comparison table (Activation Function × Optimizer)")
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
    
    # Print best performance
    print("\n" + "="*100)
    print("Best performance of each optimizer (sorted by standard L2 error | STAR-PINN)")
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
                f"  Rank {i}: {res['optimizer']:<20} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_fisher_kpp_STAR_PINN()