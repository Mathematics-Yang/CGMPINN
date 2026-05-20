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

# define activation function
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# define equation parameters
ALPHA1 = 1.0    # spatial oscillation frequency parameter
ALPHA2 = 1.0    # temporal oscillation frequency parameter
GAMMA = 0.1     # exponential decay coefficient
BETA = GAMMA    # damping coefficient 
c = np.sqrt(GAMMA**2 + (ALPHA2 * np.pi)**2) / (ALPHA1 * np.pi) # wave speed

# 1. PINN model
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation):
        super(PINN, self).__init__()
        self.activation = activation  
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, t], dim=1)  
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor

    def compute_gradients(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        return u_t, u_tt, u_x, u_xx
    
    def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_t, u_tt, _, u_xx = self.compute_gradients(x, t)
        pde_residual = u_tt + 2 * BETA * u_t - (c ** 2) * u_xx
        return torch.mean(pde_residual ** 2)
    
    def initial_loss(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        """
        Initial conditions for the wave equation:
        1. u(x, 0) = u0(x) = e^(-γ*0) * sin(α1πx) * cos(α2π*0) = sin(α1πx)
        2. u_t(x, 0) = v0(x) = -γ sin(α1πx) cos(0) - α2π sin(α1πx) sin(0) = -γ sin(α1πx)
        """
        # First initial condition: u(x,0) = sin(α1πx)
        u_pred = self.forward(x, t0)
        u0_exact = torch.sin(ALPHA1 * np.pi * x)
        initial_residual1 = u_pred - u0_exact
        
        # Second initial condition: u_t(x,0) = -γ sin(α1πx)
        u_t, _, _, _ = self.compute_gradients(x, t0)
        v0_exact = -GAMMA * torch.sin(ALPHA1 * np.pi * x)
        initial_residual2 = u_t - v0_exact
        
        # Weighted sum of the two initial condition losses
        return torch.mean(initial_residual1 ** 2) + torch.mean(initial_residual2 ** 2)
    
    # Define boundary condition loss function
    def boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        x0 = torch.zeros_like(t)
        u0_pred = self.forward(x0, t)
        u0_exact = torch.exp(-GAMMA * t) * torch.sin(ALPHA1 * np.pi * x0) * torch.cos(ALPHA2 * np.pi * t)
        x1 = torch.ones_like(t)
        u1_pred = self.forward(x1, t)
        u1_exact = torch.exp(-GAMMA * t) * torch.sin(ALPHA1 * np.pi * x1) * torch.cos(ALPHA2 * np.pi * t)
        boundary_residual1 = u0_pred - u0_exact
        boundary_residual2 = u1_pred - u1_exact
        return torch.mean(boundary_residual1 ** 2) + torch.mean(boundary_residual2 ** 2)
    
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

# 3. Training function (Adam)
def train_with_optimizer(
    model: PINN,
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

# 4. Training function (L-BFGS)
def train_with_lbfgs(
    model: PINN,
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

# 5. Two-stage training function (Adam→L-BFGS)
def train_adam_lbfgs(
    model: PINN,
    pde_data: tuple,
    initial_data: tuple,
    boundary_data: torch.Tensor,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    """Two-stage training: first Adam for global exploration, then L-BFGS for local refinement"""
    print("\n=== Stage 1: Adam Global Exploration ===")
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    train_with_optimizer(
        model=model,
        optimizer=optimizer_adam,
        epochs=adam_epochs,
        pde_data=pde_data,
        initial_data=initial_data,
        boundary_data=boundary_data,
        loss_history=loss_history
    )
    
    print("\n=== Stage 2: L-BFGS Local Refinement ===")
    train_with_lbfgs(
        model=model,
        pde_data=pde_data,
        initial_data=initial_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        lr=lr_lbfgs,
        max_iter=lbfgs_max_iter
    )

# 6. Evaluation function
def evaluate_model(
    model: PINN,
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
        fig, axes = plt.subplots(len(optim_labels), 3, figsize=(20, 5*len(optim_labels)))
        fig.suptitle(f'Solution Comparison - Activation: {activation_name}', fontsize=24, y=0.99)
        
        if len(optim_labels) == 1:
            axes = axes.reshape(1, -1)
        
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
            im3 = ax3.pcolormesh(t_grid, x_grid, error_grid, cmap=cm.RdBu_r, shading='gouraud')
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
        
        # First column: spatial waveforms at different time points
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
        
        # Plot decay envelope
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

# 8. Main function
def train_1D_wave_PINN_with_activations():
    # 1. Hyperparameter settings
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
    epochs_adam = 15000    
    max_iter_lbfgs = 15000    
    adam_lbfgs_adam_epochs = 5000  
    adam_lbfgs_lbfgs_iter = 10000  
    
    # Activation functions dictionary
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (PINN)', 'L-BFGS (PINN)', 'Adam→L-BFGS (PINN)']
    
    # 2. Generate data
    print("Generating training and testing data...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde,
        n_initial=n_initial,
        n_boundary=n_boundary,
        n_test=n_test
    )
    x_test, t_test = test_data
    
    # Calculate exact solution: u(x,t) = e^(-γt) * sin(α1πx) * cos(α2πt)
    u_exact = torch.exp(-GAMMA * t_test) * torch.sin(ALPHA1 * np.pi * x_test) * torch.cos(ALPHA2 * np.pi * t_test)
    u_exact_np = u_exact.numpy()
    u_exact_grid = u_exact_np.reshape(n_test, n_test)
    x_grid = x_test.reshape(n_test, n_test)
    t_grid = t_test.reshape(n_test, n_test)
    
    # 3. Initialize storage variables
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name} | Equation parameters: α1={ALPHA1}, α2={ALPHA2}, γ={GAMMA}, β={BETA}, c={c}")
        print("="*80)
        
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Adam training
        print(f"\n--- {activation_name} + Adam ---")
        model_adam = PINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        start_time = time.time()
        train_with_optimizer(
            model=model_adam,
            optimizer=optimizer_adam,
            epochs=epochs_adam,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam (PINN)'] = {
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
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 L-BFGS training
        print(f"\n--- {activation_name} + L-BFGS ---")
        model_lbfgs = PINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_lbfgs = []
        start_time = time.time()
        train_with_lbfgs(
            model=model_lbfgs,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_lbfgs,
            lr=lr_lbfgs_config[activation_name],
            max_iter=max_iter_lbfgs
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate L-BFGS
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['L-BFGS (PINN)'] = {
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
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Adam→L-BFGS training
        print(f"\n--- {activation_name} + Adam→L-BFGS ---")
        model_adam_lbfgs = PINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_adam_lbfgs = []
        start_time = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lr_lbfgs=lr_adam_lbfgs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam→L-BFGS
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam→L-BFGS (PINN)'] = {
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
        
        if activation_name == 'Tanh':  
            save_path = "1D_wave_tanh_adam2lbfgs_pinn.pth"
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
            print(f"✅ Tanh+Adam→LBFGS wave equation model saved to: {save_path}")
        
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save current activation function results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # 5. Visualization of results
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    plot_damping_effect(x_grid, t_grid,
                        {optim_label: activation_results['Tanh'][optim_label]['u_pred'] for optim_label in optim_labels},
                        u_exact_np)
    
    # 6. Print comprehensive comparison table
    print("\n" + "="*120)
    print("Comprehensive comparison table for all activation functions × optimizers (1D wave equation with damping)")
    print("="*120)
    header = (f"{'Activation':<10} {'Optimizer':<15} {'Training Time(s)':<12} {'L2 Error':<12} "
              f"{'L2 Relative Error':<12} {'L∞ Error':<12} {'Final Total Loss':<12}")
    print(header)
    print("-"*120)
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
                f"{results['final_total_loss']:<12.2e}"
            )
    
    # 7. Print best performance for each activation function
    print("\n" + "="*100)
    print("Best performance for each activation function (sorted by standard L2 error)")
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
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<12} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_wave_PINN_with_activations()