import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm  
import matplotlib.pyplot as plt
import time

torch.manual_seed(1224)
np.random.seed(1224)

class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# defining constants for the heat equation
ALPHA1 = 1.0  
ALPHA2 = 2.0  
k = 10.0 

# defining lightweight PINN block (basic module of STAR-PINN)
class LightweightPINNBlock(nn.Module):
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

# 1. defining STAR-PINN model (stacked adaptive residual structure)
class STAR_PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers_per_block, activation, n_blocks=3):
        super(STAR_PINN, self).__init__()
        self.n_blocks = n_blocks  # number of stacked PINN blocks
        self.activation = activation
        
        # create stacked lightweight PINN blocks
        self.pinn_blocks = nn.ModuleList([
            LightweightPINNBlock(input_dim, output_dim, hidden_dim, n_layers_per_block, activation)
            for _ in range(n_blocks)
        ])
        
        # adaptive residual weights (trainable)
        self.adaptive_weights = nn.Parameter(torch.ones(n_blocks - 1) * 0.5)  # α1, α2,...α(n_blocks-1)
    
    # forward propagation
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        forward propagation: stacked adaptive residual fusion + input x,t concatenation
        A_z^0 = PINN0([x,t])
        A_z^1 = PINN1([x,t]) + α1*A_z^0
        A_z^2 = PINN2([x,t]) + α2*A_z^1
        ...
        final output = fusion result of the last block
        """
        # concatenate x and t as input
        input_tensor = torch.cat([x, t], dim=1)
        
        # output of the first block
        current_output = self.pinn_blocks[0](input_tensor)
        
        # subsequent blocks: residual fusion
        for i in range(1, self.n_blocks):
            block_output = self.pinn_blocks[i](input_tensor)
            # adaptive weight fusion (weights constrained in [0,1] range)
            alpha = torch.sigmoid(self.adaptive_weights[i-1])  # ensure weights are non-negative and reasonable
            current_output = block_output + alpha * current_output
        
        return current_output

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
        return u_t, u_xx, u_x

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

    def pde_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u_t, u_xx, _ = self.compute_gradients(x, t)
        f = self.source_term(x, t)
        pde_residual = u_t - u_xx - f  
        return torch.mean(pde_residual ** 2)
    
    def initial_loss(self, x: torch.Tensor, t0: torch.Tensor) -> torch.Tensor:
        u_pred = self.forward(x, t0)
        u_exact = (torch.sin(ALPHA1 * np.pi * x) + torch.tanh(k * x)) * torch.sin(ALPHA2 * np.pi * t0)
        initial_residual = u_pred - u_exact
        return torch.mean(initial_residual ** 2)
    
    def boundary_loss(self, t: torch.Tensor) -> torch.Tensor:
        x0 = torch.zeros_like(t)
        u0_pred = self.forward(x0, t)
        u0_exact = (torch.sin(ALPHA1 * np.pi * x0) + torch.tanh(k * x0)) * torch.sin(ALPHA2 * np.pi * t)
        
        x1 = torch.ones_like(t)
        u1_pred = self.forward(x1, t)
        u1_exact = (torch.sin(ALPHA1 * np.pi * x1) + torch.tanh(k * x1)) * torch.sin(ALPHA2 * np.pi * t)
        
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

# 2. generate training and testing data (domain Ω=[0,1], t∈[0,1])
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

# 3. training function (Adam)
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
                f'PDE Loss: {pde_loss.item():.2e} | Initial Loss: {initial_loss.item():.2e} | '
                f'Boundary Loss: {boundary_loss.item():.2e}'
            )

# 4. training function (L-BFGS)
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
                f'PDE Loss: {pde_loss_val:.2e} | Initial Loss: {initial_loss_val:.2e} | '
                f'Boundary Loss: {boundary_loss_val:.2e}'
            )

# 5. two-stage training function (Adam→L-BFGS)
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

# 6. evaluation function (with multiple error metrics)
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

# 7. visualization function 
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
            
            # Reshape to grid
            u_pred_grid = u_pred.reshape(n_test, n_test)
            u_exact_grid = u_exact_np.reshape(n_test, n_test)
            error_grid = np.abs(u_pred_grid - u_exact_grid)
            x_grid = x_test.reshape(n_test, n_test)
            t_grid = t_test.reshape(n_test, n_test)
            
            # Predicted solution
            ax1 = axes[idx, 0]
            im1 = ax1.pcolormesh(t_grid, x_grid, u_pred_grid, cmap=cm.jet, shading='gouraud')
            ax1.set_xlabel('Time (t)', fontsize=10)
            ax1.set_ylabel('Position (x)', fontsize=10)
            ax1.set_title(f'{optim_label} - Predicted', fontsize=12)
            plt.colorbar(im1, ax=ax1, shrink=0.6, aspect=5)
            
            # Exact solution
            ax2 = axes[idx, 1]
            im2 = ax2.pcolormesh(t_grid, x_grid, u_exact_grid, cmap=cm.jet, shading='gouraud')
            ax2.set_title('Analytical Solution', fontsize=12)
            plt.colorbar(im2, ax=ax2, shrink=0.6, aspect=5)
            ax2.set_xlabel('Time (t)', fontsize=10)
            ax2.set_ylabel('Position (x)', fontsize=10)
            
            # Absolute error
            ax3 = axes[idx, 2]
            im3 = ax3.pcolormesh(t_grid, x_grid, error_grid, cmap=cm.viridis, shading='gouraud')
            ax3.set_xlabel('Time (t)', fontsize=10)
            ax3.set_ylabel('Position (x)', fontsize=10)
            ax3.set_title(f'{optim_label} - Absolute Error', fontsize=12)
            plt.colorbar(im3, ax=ax3, shrink=0.6, aspect=5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

# 8. main function
def train_1D_heat_STAR_PINN_with_activations_and_source_term():
    # 1. hyperparameter settings
    n_layers_per_block = 2    
    input_dim = 2             
    output_dim = 1           
    hidden_dim = 50           
    n_pde = 1500              
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
    optim_labels = ['Adam (STAR-PINN)', 'L-BFGS (STAR-PINN)', 'Adam→L-BFGS (STAR-PINN)']
    
    # 2. Generate data
    print("Generating training and testing data...")
    pde_data, initial_data, boundary_data, test_data = generate_data(
        n_pde=n_pde,
        n_initial=n_initial,
        n_boundary=n_boundary,
        n_test=n_test
    )
    x_test, t_test = test_data
    
    # Calculate exact solution
    u_exact = (torch.sin(ALPHA1 * np.pi * x_test) + torch.tanh(k * x_test)) * torch.sin(ALPHA2 * np.pi * t_test)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name} | Equation parameters: α1={ALPHA1}, α2={ALPHA2}")
        print(f"STAR-PINN configuration: {3} stacked blocks × {2} hidden layers per block")
        print("="*80)
        
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Adam training
        print(f"\n--- {activation_name} + Adam ---")
        model_adam = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, activation_fn)
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
        current_results['Adam (STAR-PINN)'] = {
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
        model_lbfgs = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, activation_fn)
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
        current_results['L-BFGS (STAR-PINN)'] = {
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
        model_adam_lbfgs = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, activation_fn)
        loss_history_adam_lbfgs = []
        start_time = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs,
            pde_data=pde_data,
            initial_data=initial_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs,
            lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam→L-BFGS
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_initial, final_boundary = model_adam_lbfgs.final_total_loss(pde_data, initial_data, boundary_data)
        current_results['Adam→L-BFGS (STAR-PINN)'] = {
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
            save_path = "1D_heat_tanh_adam2lbfgs_star_pinn.pth"
    
            # Package all content to be saved
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
    
            # Save custom dictionary
            torch.save(complete_save_dict, save_path)
            print(f"✅ Tanh+Adam→L-BFGS complete information saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save current activation function results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # 5. Result visualization
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # 6. Print comprehensive comparison table   
    print("\n" + "="*120)
    print("All Activation Functions × Optimizers Comprehensive Comparison Table (Including Source Term Heat Equation - STAR-PINN)")
    print("="*120)
    header = (f"{'Activation':<10} {'Optimizer':<20} {'Training Time (s)':<12} {'Standard L2 Error':<12} "
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
    print("Best Performance of Each Activation Function (Sorted by Standard L2 Error - STAR-PINN)")
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
        # Sort in ascending order by L2 error
        optim_results_sorted = sorted(optim_results, key=lambda x: x['l2_error'])
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<20} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_heat_STAR_PINN_with_activations_and_source_term()