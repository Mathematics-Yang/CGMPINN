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

# define constants
BETA1 = 3.0  # x direction oscillation frequency parameter
BETA2 = 2.0  # y direction oscillation frequency parameter
DOMAIN_MIN = 0.0  
DOMAIN_MAX = 1.0 

# 1. gPINN model
class gPINN2D(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, grad_weight=0.01):
        super(gPINN2D, self).__init__()
        self.activation = activation  
        self.grad_weight = grad_weight  
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    # forward propagation
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, y], dim=1)  
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor

    # analytical solution
    def analytical_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sin_part = torch.sin(BETA1 * np.pi * x) * torch.sin(BETA2 * np.pi * y)
        exp_part = torch.exp(-x**2 - y**2)
        return sin_part + exp_part

    # define source term f(x,y)
    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sin_beta1pi_x = torch.sin(BETA1 * np.pi * x)
        sin_beta2pi_y = torch.sin(BETA2 * np.pi * y)
        exp_term = torch.exp(-x**2 - y**2)
        laplacian_first = - (np.pi**2) * (BETA1**2 + BETA2**2) * sin_beta1pi_x * sin_beta2pi_y
        laplacian_second = 4 * (x**2 + y**2 - 1) * exp_term
        f = laplacian_first + laplacian_second
        return f

    # compute high-order gradients (for gradient-enhanced loss)
    def compute_high_order_gradients(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        u = self.forward(x, y)  
        u_x = torch.autograd.grad(
            outputs=u, inputs=x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        u_y = torch.autograd.grad(
            outputs=u, inputs=y, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        u_yy = torch.autograd.grad(
            outputs=u_y, inputs=y, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        laplacian_u = u_xx + u_yy

        # pde residual: Δu - f(x,y)
        f_exact = self.source_term(x, y)
        f_res = laplacian_u - f_exact
        f_res_x = torch.autograd.grad(
            outputs=f_res, inputs=x, grad_outputs=torch.ones_like(f_res),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        f_res_y = torch.autograd.grad(
            outputs=f_res, inputs=y, grad_outputs=torch.ones_like(f_res),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        return laplacian_u, f_res, f_res_x, f_res_y

    def pde_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, f_res, _, _ = self.compute_high_order_gradients(x, y)
        return torch.mean(f_res ** 2)
    
    # gradient loss function 
    def grad_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Gradient loss: constrain the gradient of the PDE residual to be zero (df_res/dx = 0, df_res/dy = 0)
        Corresponding formula: L_g = w_g * (mean(df_res/dx²) + mean(df_res/dy²))
        """
        _, _, f_res_x, f_res_y = self.compute_high_order_gradients(x, y)
        grad_loss_x = torch.mean(f_res_x ** 2)
        grad_loss_y = torch.mean(f_res_y ** 2)
        return self.grad_weight * (grad_loss_x + grad_loss_y)
    
    def boundary_loss(self, boundary_data) -> torch.Tensor:
        x0, y0, x1, y1, x2, y2, x3, y3 = boundary_data
        
        u0_pred = self.forward(x0, y0)
        u0_exact = self.analytical_solution(x0, y0)
        
        u1_pred = self.forward(x1, y1)
        u1_exact = self.analytical_solution(x1, y1)
        
        u2_pred = self.forward(x2, y2)
        u2_exact = self.analytical_solution(x2, y2)
        
        u3_pred = self.forward(x3, y3)
        u3_exact = self.analytical_solution(x3, y3)
        
        boundary_loss = (torch.mean((u0_pred - u0_exact)**2) + 
                         torch.mean((u1_pred - u1_exact)**2) + 
                         torch.mean((u2_pred - u2_exact)**2) + 
                         torch.mean((u3_pred - u3_exact)**2))
        return boundary_loss
    
    def final_total_loss(self, pde_data, boundary_data) -> tuple[float, float, float, float]:
        x_pde, y_pde = pde_data
        
        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde, y_pde).item()
            boundary_loss_val = self.boundary_loss(boundary_data).item()
            grad_loss_val = self.grad_loss(x_pde, y_pde).item()
            total_loss_val = pde_loss_val + boundary_loss_val + grad_loss_val
        
        return total_loss_val, pde_loss_val, boundary_loss_val, grad_loss_val

# 2. generate training and testing data
def generate_data(
    n_pde: int,
    n_boundary: int,
    n_test: int 
) -> tuple:
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
    
    # Test data (grid points)
    x_test = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_test).reshape(-1, 1)
    y_test = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_test).reshape(-1, 1)
    x_test_grid, y_test_grid = torch.meshgrid(x_test.squeeze(), y_test.squeeze(), indexing='ij')
    x_test_flat = x_test_grid.reshape(-1, 1)
    y_test_flat = y_test_grid.reshape(-1, 1)
    test_data = (x_test_flat, y_test_flat)
    
    return pde_data, boundary_data, test_data

# 3. adam training function
def train_with_optimizer_gpinn2d(
    model: gPINN2D,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list
) -> None:
    x_pde, y_pde = pde_data
    
    model.train() 

    for epoch in range(epochs):
        pde_loss = model.pde_loss(x_pde, y_pde)
        boundary_loss = model.boundary_loss(boundary_data)
        grad_loss = model.grad_loss(x_pde, y_pde)
        total_loss = pde_loss + boundary_loss + grad_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # record loss 
        loss_history.append(total_loss.item())
        
        # Print progress every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(
                f'Epoch {epoch+1:4d} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss.item():.2e} | Boundary Loss: {boundary_loss.item():.2e} | '
                f'Gradient Loss: {grad_loss.item():.2e}'
            )

# 4. L-BFGS training function
def train_with_lbfgs_gpinn2d(
    model: gPINN2D,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    lr: float,
    max_iter: int
) -> None:
    x_pde, y_pde = pde_data
    
    model.train()  
    pde_loss_val = boundary_loss_val = grad_loss_val = 0.0

    # Define closure function (required by L-BFGS)
    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, boundary_loss_val, grad_loss_val
        optimizer.zero_grad()
        
        # Compute losses (including gradient loss + fixed boundary points)
        pde_loss = model.pde_loss(x_pde, y_pde)
        boundary_loss = model.boundary_loss(boundary_data)
        grad_loss = model.grad_loss(x_pde, y_pde)
        total_loss = pde_loss + boundary_loss + grad_loss
        
        # Backpropagation
        total_loss.backward()
        
        # Save current component values
        pde_loss_val = pde_loss.item()
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
    
    # Start training
    for iter_idx in range(max_iter):
        total_loss = optimizer.step(closure)
        current_loss = total_loss.item()
        loss_history.append(current_loss)
        
        # Print iteration information
        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            print(
                f'Iteration {iter_idx+1}/{max_iter} | Total Loss: {current_loss:.2e} | '
                f'PDE Loss: {pde_loss_val:.2e} | Boundary Loss: {boundary_loss_val:.2e} | '
                f'Gradient Loss: {grad_loss_val:.2e}'
            )

# 5. Two-stage training function (Adam→L-BFGS)
def train_adam_lbfgs_gpinn2d(
    model: gPINN2D,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    print("\n=== Stage 1: Adam Global Exploration ===")
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    train_with_optimizer_gpinn2d(
        model=model,
        optimizer=optimizer_adam,
        epochs=adam_epochs,
        pde_data=pde_data,
        boundary_data=boundary_data,
        loss_history=loss_history
    )
    
    print("\n=== Stage 2: L-BFGS Local Refinement ===")
    train_with_lbfgs_gpinn2d(
        model=model,
        pde_data=pde_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        lr=lr_lbfgs,
        max_iter=lbfgs_max_iter
    )

# 6. Evaluation function
def evaluate_model(
    model: gPINN2D,
    test_data: tuple,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
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
    x_test, y_test = test_data
    n_test = int(np.sqrt(len(x_test)))
    
    # Reshape grid
    x_grid = x_test.reshape(n_test, n_test).numpy()
    y_grid = y_test.reshape(n_test, n_test).numpy()
    u_exact_grid = u_exact_np.reshape(n_test, n_test)
    
    for activation_name, results in activation_results.items():
        fig, axes = plt.subplots(3, 3, figsize=(20, 14))
        fig.suptitle(f'Solution Comparison - Activation: {activation_name}', fontsize=24, y=0.99)
        
        for idx, optim_label in enumerate(optim_labels):
            u_pred = results[optim_label]['u_pred']
            
            # Reshape to grid form
            u_pred_grid = u_pred.reshape(n_test, n_test)
            error_grid = np.abs(u_pred_grid - u_exact_grid)
            
            # Column 1: Predicted Solution
            ax1 = axes[idx, 0]
            im1 = ax1.pcolormesh(x_grid, y_grid, u_pred_grid, cmap=cm.jet, shading='gouraud')
            ax1.set_xlabel('x', fontsize=10)
            ax1.set_ylabel('y', fontsize=10)
            ax1.set_title(f'{optim_label} - Predicted', fontsize=12)
            plt.colorbar(im1, ax=ax1, shrink=0.6, aspect=5)
            
            # Column 2: Analytical Solution
            ax2 = axes[idx, 1]
            im2 = ax2.pcolormesh(x_grid, y_grid, u_exact_grid, cmap=cm.jet, shading='gouraud')
            ax2.set_title('Analytical Solution', fontsize=12)
            plt.colorbar(im2, ax=ax2, shrink=0.6, aspect=5)
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('y', fontsize=10)
            
            # Column 3: Absolute Error
            ax3 = axes[idx, 2]
            im3 = ax3.pcolormesh(x_grid, y_grid, error_grid, cmap=cm.viridis, shading='gouraud')
            ax3.set_xlabel('x', fontsize=10)
            ax3.set_ylabel('y', fontsize=10)
            ax3.set_title(f'{optim_label} - Absolute Error', fontsize=12)
            plt.colorbar(im3, ax=ax3, shrink=0.6, aspect=5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

# 8. main function
def train_2D_poisson_gPINN_with_activations_and_source_term():
    # 1. hyperparameters and configurations
    n_layers = 4
    input_dim = 2
    output_dim = 1
    hidden_dim = 50
    grad_weight = 0.01  # gPINN gradient loss weight
    n_pde = 2000       
    n_boundary = 250    
    n_test = 100
    lr_lbfgs_config = {'Tanh': 1.0}
    epochs_adam = 15000
    lr_adam_lbfgs = 0.8
    max_iter_lbfgs = 15000
    adam_lbfgs_adam_epochs = 5000
    adam_lbfgs_lbfgs_iter = 10000
    
    # Define activation functions dictionary
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    
    # Optimizer labels 
    optim_labels = ['Adam (gPINN)', 'L-BFGS (gPINN)', 'Adam→L-BFGS (gPINN)']
    
    # 2. Generate data 
    print("Generating training and testing data...")
    pde_data, boundary_data, test_data = generate_data(
        n_pde=n_pde,
        n_boundary=n_boundary,
        n_test=n_test
    )
    x_test, y_test = test_data
    
    # Calculate exact solution
    model_temp = gPINN2D(input_dim, output_dim, hidden_dim, 1, TanhActivation())
    u_exact = model_temp.analytical_solution(x_test, y_test)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation Function: {activation_name} | Equation Parameters: β1={BETA1}, β2={BETA2}")
        print("="*80)
        
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Train Adam (gPINN2D)
        print(f"\n--- {activation_name} + Adam (gPINN) ---")
        model_adam = gPINN2D(input_dim, output_dim, hidden_dim, n_layers, activation_fn, grad_weight=grad_weight)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        start_time_adam = time.time()
        train_with_optimizer_gpinn2d(
            model=model_adam,
            optimizer=optimizer_adam,
            epochs=epochs_adam,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam
        )
        training_time_adam = time.time() - start_time_adam
        current_training_times.append(training_time_adam)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_boundary, final_grad = model_adam.final_total_loss(pde_data, boundary_data)
        current_results['Adam (gPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'final_grad_loss': final_grad,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam)
        print(f"Adam training time: {training_time_adam:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 Train L-BFGS (gPINN2D)
        print(f"\n--- {activation_name} + L-BFGS (gPINN) ---")
        model_lbfgs = gPINN2D(input_dim, output_dim, hidden_dim, n_layers, activation_fn, grad_weight=grad_weight)
        loss_history_lbfgs = []
        start_time_lbfgs = time.time()
        train_with_lbfgs_gpinn2d(
            model=model_lbfgs,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_lbfgs,
            lr=lr_lbfgs_config[activation_name],
            max_iter=max_iter_lbfgs
        )
        training_time_lbfgs = time.time() - start_time_lbfgs
        current_training_times.append(training_time_lbfgs)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary, final_grad = model_lbfgs.final_total_loss(pde_data, boundary_data)
        current_results['L-BFGS (gPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'final_grad_loss': final_grad,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_lbfgs)
        print(f"L-BFGS training time: {training_time_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Train Adam→L-BFGS (gPINN2D)
        print(f"\n--- {activation_name} + Adam→L-BFGS (gPINN) ---")
        model_adam_lbfgs = gPINN2D(input_dim, output_dim, hidden_dim, n_layers, activation_fn, grad_weight=grad_weight)
        loss_history_adam_lbfgs = []
        start_time_adam_lbfgs = time.time()
        train_adam_lbfgs_gpinn2d(
            model=model_adam_lbfgs,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs,
            lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        training_time_adam_lbfgs = time.time() - start_time_adam_lbfgs
        current_training_times.append(training_time_adam_lbfgs)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary, final_grad = model_adam_lbfgs.final_total_loss(pde_data, boundary_data)
        current_results['Adam→L-BFGS (gPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'final_grad_loss': final_grad,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        if activation_name == 'Tanh':  
            save_path = "2D_poisson_tanh_adam2lbfgs_gpinn.pth"
            complete_save_dict = {
                'model_state_dict': model_adam_lbfgs.state_dict(),  
                'loss_history': loss_history_adam_lbfgs,           
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
            torch.save(complete_save_dict, save_path)
            print(f"✅ Tanh+Adam→LBFGS complete information saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {training_time_adam_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # 5. Visualization
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # 6. Print comprehensive comparison table
    print("\n" + "="*140)
    print("gPINN 2D Poisson Equation All Activation Functions × Optimizers Comprehensive Comparison Table (Fixed Boundary Points)")
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
    
    # 7. Print the best performance of each activation function
    print("\n" + "="*120)
    print("gPINN 2D Poisson Equation Best Performance of Each Activation Function (Sorted by Standard L2 Error - Fixed Boundary Points)")
    print("="*120)
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
                'final_loss': results['final_total_loss'],
                'final_grad_loss': results['final_grad_loss']
            })
        # sort results by L2 error
        optim_results_sorted = sorted(optim_results, key=lambda x: x['l2_error'])
        # Print sorted results
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<12} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e} | Final Gradient Loss: {res['final_grad_loss']:.2e}"
            )

# Execute main function
train_2D_poisson_gPINN_with_activations_and_source_term()