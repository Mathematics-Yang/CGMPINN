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

# Fisher-KPP equation parameters
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

# Fisher-KPP Ablowitz-Zeppetella exact traveling wave solution
def fisher_kpp_exact_solution(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    z = LAMBDA * (x - c * t)
    # Numerical stability to avoid exp overflow
    z_clamped = torch.clamp(z, min=-50, max=50)
    exp_z = torch.exp(z_clamped)
    u = 1.0 / (1.0 + exp_z) ** 2  
    return u

# gPINN model
class gPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, grad_weight=0.01):
        super(gPINN, self).__init__()
        self.activation = activation  
        self.grad_weight = grad_weight  
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)
        return self.layers(input_tensor)

    def compute_high_order_gradients(self, x: torch.Tensor, t: torch.Tensor) -> tuple:
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
        
        pde_residual = u_t - D * u_xx - r * u * (1 - u)
        
        res_x = torch.autograd.grad(
            outputs=pde_residual, inputs=x, grad_outputs=torch.ones_like(pde_residual),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        res_t = torch.autograd.grad(
            outputs=pde_residual, inputs=t, grad_outputs=torch.ones_like(pde_residual),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        return u_t, u_xx, pde_residual, res_x, res_t

    def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, _, pde_residual, _, _ = self.compute_high_order_gradients(x, t)
        return torch.mean(pde_residual ** 2)
    
    def grad_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, _, _, res_x, res_t = self.compute_high_order_gradients(x, t)
        grad_loss_x = torch.mean(res_x ** 2)
        grad_loss_t = torch.mean(res_t ** 2)
        return self.grad_weight * (grad_loss_x + grad_loss_t)
    
    def initial_loss(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u_exact = fisher_kpp_exact_solution(x, t0)
        return torch.mean((u_pred - u_exact) ** 2)
    
    def boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        x_left = torch.full_like(t, x_min, requires_grad=True)
        u_left_pred = self.forward(x_left, t)
        u_left_exact = fisher_kpp_exact_solution(x_left, t)
        
        x_right = torch.full_like(t, x_max, requires_grad=True)
        u_right_pred = self.forward(x_right, t)
        u_right_exact = fisher_kpp_exact_solution(x_right, t)
        
        return torch.mean((u_left_pred - u_left_exact) ** 2) + torch.mean((u_right_pred - u_right_exact) ** 2)
    
    def final_total_loss(self, pde_data, initial_data, boundary_data) -> tuple[float, float, float, float, float]:
        x_pde, t_pde = pde_data
        x_initial, t_initial = initial_data
        t_boundary = boundary_data
        
        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde, t_pde).item()
            initial_loss_val = self.initial_loss(x_initial, t_initial).item()
            boundary_loss_val = self.boundary_loss(t_boundary).item()
            grad_loss_val = self.grad_loss(x_pde, t_pde).item()  
            total_loss_val = pde_loss_val + initial_loss_val + boundary_loss_val + grad_loss_val
        
        return total_loss_val, pde_loss_val, initial_loss_val, boundary_loss_val, grad_loss_val

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

# training functions
def train_with_optimizer_gpinn(
    model: gPINN,
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
        grad_loss = model.grad_loss(x_pde, t_pde)
        total_loss = pde_loss + initial_loss + boundary_loss + grad_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(
                f'Epoch {epoch+1:4d} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss.item():.2e} | Initial Loss: {initial_loss.item():.2e} | '
                f'Boundary Loss: {boundary_loss.item():.2e} | Gradient Loss: {grad_loss.item():.2e}'
            )

def train_with_lbfgs_gpinn(
    model: gPINN,
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

    # save loss components for logging
    pde_loss_val = initial_loss_val = boundary_loss_val = grad_loss_val = 0.0

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, initial_loss_val, boundary_loss_val, grad_loss_val
        optimizer.zero_grad()
        pde_loss = model.pde_loss(x_pde, t_pde)
        initial_loss = model.initial_loss(x_initial, t_initial)
        boundary_loss = model.boundary_loss(t_boundary)
        grad_loss = model.grad_loss(x_pde, t_pde)
        total_loss = pde_loss + initial_loss + boundary_loss + grad_loss
        total_loss.backward()
        pde_loss_val = pde_loss.item()
        initial_loss_val = initial_loss.item()
        boundary_loss_val = boundary_loss.item()
        grad_loss_val = grad_loss.item()
        return total_loss
    
    # Initialize L-BFGS optimizer
    optimizer = optim.LBFGS(
        model.parameters(),
        max_iter=1,
        max_eval=10,
        line_search_fn='strong_wolfe',
        lr=lr
    )
    
    # L-BFGS iterative training
    for iter_idx in range(max_iter):
        total_loss = optimizer.step(closure)
        loss_history.append(total_loss.item())
        # Print progress
        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            print(
                f'Iteration {iter_idx+1}/{max_iter} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss_val:.2e} | Initial Loss: {initial_loss_val:.2e} | '
                f'Boundary Loss: {boundary_loss_val:.2e} | Gradient Loss: {grad_loss_val:.2e}'
            )

def train_adam_lbfgs_gpinn(
    model: gPINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    """Two-stage training: Adam global exploration → L-BFGS local fine-tuning (gPINN version)"""
    print("\n=== Stage 1: Adam Global Exploration ===")
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    train_with_optimizer_gpinn(
        model=model, optimizer=optimizer_adam, epochs=adam_epochs,
        pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
        loss_history=loss_history
    )
    
    print("\n=== Stage 2: L-BFGS Local Fine-tuning ===")
    train_with_lbfgs_gpinn(
        model=model, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
        loss_history=loss_history, lr=lr_lbfgs, max_iter=lbfgs_max_iter
    )

# Evaluation function
def evaluate_model(
    model: gPINN,
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
        plt.title(f'Fisher-KPP gPINN Training Loss (Activation: {activation_name})', fontsize=14)
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
        fig.suptitle(f'Fisher-KPP gPINN Solution Comparison (Activation: {activation_name})', fontsize=24, y=0.99)
        
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
def train_1D_fisher_kpp_gPINN():
    # Hyperparameter settings
    n_layers = 4               
    input_dim = 2              
    output_dim = 1             
    hidden_dim = 80
    grad_weight = 0.01         
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

    # Activation functions + optimizer labels
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (gPINN)', 'L-BFGS (gPINN)', 'Adam→L-BFGS (gPINN)']
    
    # Generate Fisher-KPP training/testing data
    print("Generating Fisher-KPP training/testing data...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde, n_initial=n_initial, n_boundary=n_boundary, n_test=n_test
    )
    x_test, t_test = test_data
    # Compute exact solution for test set
    u_exact = fisher_kpp_exact_solution(x_test, t_test)
    u_exact_np = u_exact.numpy()
    
    # Initialize result storage
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name} | Fisher-KPP parameters: D={D}, r={r}, c={c:.2f}")
        print("="*80)
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 1. Pure Adam training (gPINN version)
        print(f"\n--- {activation_name} + Adam (gPINN) ---")
        model_adam = gPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn, grad_weight)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        start_time = time.time()
        train_with_optimizer_gpinn(
            model=model_adam, optimizer=optimizer_adam, epochs=epochs_adam,
            pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        # Evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary, final_grad = model_adam.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam (gPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'final_grad_loss': final_grad, 'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 2. Pure L-BFGS training (gPINN version)
        print(f"\n--- {activation_name} + L-BFGS (gPINN) ---")
        model_lbfgs = gPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn, grad_weight)
        loss_history_lbfgs = []
        start_time = time.time()
        train_with_lbfgs_gpinn(
            model=model_lbfgs, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_lbfgs, lr=lr_lbfgs_config[activation_name], max_iter=max_iter_lbfgs
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        # Evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary, final_grad = model_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['L-BFGS (gPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'final_grad_loss': final_grad, 'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 3. Adam→L-BFGS two-stage training (gPINN best model, save)
        print(f"\n--- {activation_name} + Adam→L-BFGS (gPINN) ---")
        model_adam_lbfgs = gPINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn, grad_weight)
        loss_history_adam_lbfgs = []
        start_time = time.time()
        train_adam_lbfgs_gpinn(
            model=model_adam_lbfgs, pde_data=pde_data, initial_data=initial_data, boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs, adam_epochs=adam_lbfgs_adam_epochs,
            lr_lbfgs=lr_adam_lbfgs, lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        # Evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary, final_grad = model_adam_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam→L-BFGS (gPINN)'] = {
            'l2_error': l2_err, 'l2_relative_error': l2_rel_err, 'linf_error': linf_err,
            'final_total_loss': final_total, 'final_pde_loss': final_pde,
            'final_initial_loss': final_initial, 'final_boundary_loss': final_boundary,
            'final_grad_loss': final_grad, 'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        
        # Save gPINN best model (Tanh+Adam→L-BFGS)
        save_path = "1D_fisher_kpp_tanh_adam2lbfgs_gpinn.pth"
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
        print(f"✅ Fisher-KPP gPINN best model saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save all results for the current activation function
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # Visualize results
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # Print comprehensive comparison table for gPINN (with added gradient loss column)
    print("\n" + "="*140)
    print("Fisher-KPP equation gPINN training comprehensive comparison table (activation function × optimizer)")
    print("="*140)
    header = (f"{'Activation':<10} {'Optimizer':<15} {'Training Time(s)':<12} {'Standard L2 Error':<12} "
              f"{'L2 Relative Error':<12} {'L∞ Error':<12} {'Final Total Loss':<12} {'Final Gradient Loss':<12}")
    print(header)
    print("-"*140)
    for activation_name in activation_names:
        for idx, optim_label in enumerate(optim_labels):
            results = activation_results[activation_name][optim_label]
            train_time = activation_training_times[activation_name][idx]
            print(
                f"{activation_name:<10} "
                f"{optim_label:<15} "
                f"{train_time:<12.2f} "
                f"{results['l2_error']:<12.2e} "
                f"{results['l2_relative_error']:<12.2e} "
                f"{results['linf_error']:<12.2e} "
                f"{results['final_total_loss']:<12.2e} "
                f"{results['final_grad_loss']:<12.2e}"
            )
    
    # Print best performance ranking for gPINN
    print("\n" + "="*120)
    print("Fisher-KPP gPINN best performance of each optimizer (sorted by standard L2 error)")
    print("="*120)
    for activation_name in activation_names:
        print(f"\n[{activation_name}] Best gPINN Performance Ranking:")
        optim_results = []
        for optim_label, results in activation_results[activation_name].items():
            train_time = activation_training_times[activation_name][optim_labels.index(optim_label)]
            optim_results.append({
                'optimizer': optim_label, 'train_time': train_time,
                'l2_error': results['l2_error'], 'linf_error': results['linf_error'],
                'final_loss': results['final_total_loss'], 'final_grad_loss': results['final_grad_loss']
            })
        optim_results_sorted = sorted(optim_results, key=lambda x: x['l2_error'])
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<12} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e} | Final Gradient Loss: {res['final_grad_loss']:.2e}"
            )

# Execute main function
train_1D_fisher_kpp_gPINN()