import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm  
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

torch.manual_seed(1224)
np.random.seed(1224)

class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# define constants
BETA1 = 3.0  
BETA2 = 2.0  
DOMAIN_MIN = 0.0  
DOMAIN_MAX = 1.0 

# 1. lbPINN model
class lbPINN2D(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation):
        super(lbPINN2D, self).__init__()
        self.activation = activation 
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        
        # initialize adaptive weighting parameters (for PDE and boundary condition losses)
        # use log variance parameterization for numerical stability
        self.log_var_pde = nn.Parameter(torch.tensor(0.0))    # log variance for PDE loss
        self.log_var_boundary = nn.Parameter(torch.tensor(0.0))# log variance for boundary loss

        # regularization coefficient
        self.reg_coeff = 0.5

    # forward propagation
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input_tensor = torch.cat([x, y], dim=1)  
        for layer in self.layers:
            input_tensor = layer(input_tensor)
        return input_tensor

    def compute_laplacian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = self.forward(x, y)  
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
        u_y = torch.autograd.grad(
            outputs=u,
            inputs=y,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        u_yy = torch.autograd.grad(
            outputs=u_y,
            inputs=y,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        laplacian_u = u_xx + u_yy
        return laplacian_u

    # define source term f(x,y)
    def source_term(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sin_beta1pi_x = torch.sin(BETA1 * np.pi * x)
        sin_beta2pi_y = torch.sin(BETA2 * np.pi * y)
        exp_term = torch.exp(-x**2 - y**2)
        laplacian_first = - (np.pi**2) * (BETA1**2 + BETA2**2) * sin_beta1pi_x * sin_beta2pi_y
        laplacian_second = 4 * (x**2 + y**2 - 1) * exp_term
        f = laplacian_first + laplacian_second
        return f

    # analytical solution
    def analytical_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sin_part = torch.sin(BETA1 * np.pi * x) * torch.sin(BETA2 * np.pi * y)
        exp_part = torch.exp(-x**2 - y**2)
        return sin_part + exp_part

    def pde_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        laplacian_u = self.compute_laplacian(x, y)
        f = self.source_term(x, y)
        pde_residual = laplacian_u - f  
        return pde_residual
    
    def boundary_residual(self, boundary_data) -> torch.Tensor:
        x0, y0, x1, y1, x2, y2, x3, y3 = boundary_data
        
        u0_pred = self.forward(x0, y0)
        u0_exact = self.analytical_solution(x0, y0)
        
        u1_pred = self.forward(x1, y1)
        u1_exact = self.analytical_solution(x1, y1)
        
        u2_pred = self.forward(x2, y2)
        u2_exact = self.analytical_solution(x2, y2)
        
        u3_pred = self.forward(x3, y3)
        u3_exact = self.analytical_solution(x3, y3)
        
        boundary_residual = torch.cat([
            u0_pred - u0_exact,
            u1_pred - u1_exact,
            u2_pred - u2_exact,
            u3_pred - u3_exact
        ], dim=0)
        
        return boundary_residual
    
    # adaptive loss function 
    def adaptive_loss(self, pde_data, boundary_data) -> torch.Tensor:
        """
        adaptive loss function: based on Gaussian probability model and maximum likelihood estimation
        loss = weighted sum of residuals + regularization term (to prevent extreme weights)
        """
        x_pde, y_pde = pde_data
        
        # compute residuals
        pde_res = self.pde_residual(x_pde, y_pde)
        boundary_res = self.boundary_residual(boundary_data)
        
        # compute raw adaptive weights (1/(2σ²) = 0.5*exp(-log_var))
        raw_weight_pde = 0.5 * torch.exp(-self.log_var_pde)
        raw_weight_boundary = 0.5 * torch.exp(-self.log_var_boundary)

        # weighted residual losses
        loss_pde = raw_weight_pde * torch.mean(pde_res ** 2)
        loss_boundary = raw_weight_boundary * torch.mean(boundary_res ** 2)
        
        # regularization term to prevent extreme weights
        reg_term = self.reg_coeff * (F.softplus(self.log_var_pde) + F.softplus(self.log_var_boundary))
        
        # total loss
        total_loss = loss_pde + loss_boundary + reg_term
        return total_loss, loss_pde, loss_boundary
    
    # compute final total loss (for evaluation)
    def final_total_loss(self, pde_data, boundary_data) -> tuple[float, float, float]:
        with torch.enable_grad():
            total_loss, pde_loss, boundary_loss = self.adaptive_loss(pde_data, boundary_data)
            return (total_loss.item(), pde_loss.item(), boundary_loss.item())
    
    # get current adaptive weights (for monitoring)
    def get_adaptive_weights(self) -> dict:
        raw_weight_pde = 0.5 * torch.exp(-self.log_var_pde).item()
        raw_weight_boundary = 0.5 * torch.exp(-self.log_var_boundary).item()
        raw_log_var_pde = self.log_var_pde.item()
        raw_log_var_boundary = self.log_var_boundary.item()

        return {
            'raw_weight_pde': raw_weight_pde,
            'raw_weight_boundary': raw_weight_boundary,
            'raw_log_var_pde': raw_log_var_pde,
            'raw_log_var_boundary': raw_log_var_boundary,
            'raw_weight_sum': raw_weight_pde + raw_weight_boundary,
            'total': 0.0
        }

# 2. generate training and test data
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
    
    x_test = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_test).reshape(-1, 1)
    y_test = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, n_test).reshape(-1, 1)
    x_test_grid, y_test_grid = torch.meshgrid(x_test.squeeze(), y_test.squeeze(), indexing='ij')
    x_test_flat = x_test_grid.reshape(-1, 1)
    y_test_flat = y_test_grid.reshape(-1, 1)
    test_data = (x_test_flat, y_test_flat)
    
    return pde_data, boundary_data, test_data

# 3. Adam training function 
def train_lbpinn_with_optimizer(
    model: lbPINN2D,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    weight_history: list
) -> None:
    model.train()
    for epoch in range(epochs):
        x_pde, y_pde = pde_data
        pde_res = model.pde_residual(x_pde, y_pde)
        boundary_res = model.boundary_residual(boundary_data)
        
        # compute raw adaptive weights
        raw_weight_pde = 0.5 * torch.exp(-model.log_var_pde)
        raw_weight_boundary = 0.5 * torch.exp(-model.log_var_boundary)
        
        # compute losses
        pde_loss = raw_weight_pde * torch.mean(pde_res ** 2)
        boundary_loss = raw_weight_boundary * torch.mean(boundary_res ** 2)
        reg_term = model.reg_coeff * (F.softplus(model.log_var_pde) + F.softplus(model.log_var_boundary))
        total_loss = pde_loss + boundary_loss + reg_term
        
        # backward propagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # record losses
        loss_history.append({
            'total': total_loss.item(),
            'pde': pde_loss.item(),
            'boundary': boundary_loss.item()
        })
        
        # record weights
        if epoch % 5 == 0:
            weights = model.get_adaptive_weights()
            weights['total'] = total_loss.item()  
            weight_history.append(weights)
        
        # print information
        if (epoch + 1) % 1000 == 0:
            weights = model.get_adaptive_weights()
            print(
                f'Epoch {epoch+1:4d} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss.item():.2e} | Boundary Loss: {boundary_loss.item():.2e} '
                f'| Weights: [PDE: {weights["raw_weight_pde"]:.2e}, Boundary: {weights["raw_weight_boundary"]:.2e}]'
            )

# 4. L-BFGS training function
def train_lbfgs_lbpinn(
    model: lbPINN2D,
    pde_data: tuple,
    boundary_data: tuple,
    loss_history: list,
    weight_history: list,  
    lbfgs_max_iter: int,   
    lr_lbfgs: float      
) -> None:
    x_pde, y_pde = pde_data
    
    model.train()  
    pde_loss_val = boundary_loss_val = 0.0

    # Define closure function (required by L-BFGS)
    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, boundary_loss_val
        
        optimizer.zero_grad()  
        
        # 1. Compute residuals (using fixed boundary points)
        pde_res = model.pde_residual(x_pde, y_pde)
        boundary_res = model.boundary_residual(boundary_data)
        
        # 2. Compute raw adaptive weights
        raw_weight_pde = 0.5 * torch.exp(-model.log_var_pde)
        raw_weight_boundary = 0.5 * torch.exp(-model.log_var_boundary)
        
        # 3. Compute component losses
        pde_loss = raw_weight_pde * torch.mean(pde_res ** 2)
        boundary_loss = raw_weight_boundary * torch.mean(boundary_res ** 2)
        
        # 4. Regularization term
        reg_term = model.reg_coeff * (
            F.softplus(model.log_var_pde) + F.softplus(model.log_var_boundary)
        )
        
        # 5. Total loss
        total_loss = pde_loss + boundary_loss + reg_term
        
        # 6. Backward propagation
        total_loss.backward()
        
        # 7. Save current component loss values (for printing)
        pde_loss_val = pde_loss.item()
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
                f'PDE Loss: {pde_loss_val:.2e} | Boundary Loss: {boundary_loss_val:.2e} | '
                f'Weights: [PDE: {weights["raw_weight_pde"]:.2e}, Boundary: {weights["raw_weight_boundary"]:.2e}]'
            )

# 5. two-stage training function
def train_adam_lbfgs_lbpinn(
    model: lbPINN2D,
    pde_data: tuple,
    boundary_data: tuple,
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
        boundary_data=boundary_data,
        loss_history=loss_history,
        weight_history=weight_history
    )
    # L-BFGS optimizer phase
    print("\n=== Using L-BFGS optimizer for fine-tuning ===")
    train_lbfgs_lbpinn(
        model=model,
        pde_data=pde_data,
        boundary_data=boundary_data,
        loss_history=loss_history,
        weight_history=weight_history,
        lbfgs_max_iter=lbfgs_max_iter,
        lr_lbfgs=lr_lbfgs
    )

# 6. Evaluation function
def evaluate_model(
    model: lbPINN2D,
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
def plot_adaptive_weights(weight_histories: dict, optim_labels: list) -> None:
    for activation_name, optim_weights_list in weight_histories.items():
        for opt_idx, (weights_list, opt_label) in enumerate(zip(optim_weights_list, optim_labels)):
            if not weights_list:  
                continue
                
            plt.figure(figsize=(10, 10))
            epochs = [i*5 for i in range(len(weights_list))]
            
            # Raw weights/Log variance
            raw_weight_pde = [w['raw_weight_pde'] for w in weights_list]
            raw_weight_boundary = [w['raw_weight_boundary'] for w in weights_list]
            raw_log_var_pde = [w['raw_log_var_pde'] for w in weights_list]
            raw_log_var_boundary = [w['raw_log_var_boundary'] for w in weights_list]
            
            # 1. Raw weights
            plt.subplot(3, 1, 1)
            plt.plot(epochs, raw_weight_pde, label='raw pde weight', linewidth=2)
            plt.plot(epochs, raw_weight_boundary, label='raw boundary weight', linewidth=2)
            plt.yscale('log')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Raw Weight (Log Scale)', fontsize=12)
            plt.title(f'Raw Adaptive Weights (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
            # 2. Log-Variance
            plt.subplot(3, 1, 2)
            plt.plot(epochs, raw_log_var_pde, label='log_var pde', linewidth=2)
            plt.plot(epochs, raw_log_var_boundary, label='log_var boundary', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Log-Variance', fontsize=12)
            plt.title(f'Log-Variance Evolution (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.legend()
            
            # 3. Total Loss
            plt.subplot(3, 1, 3)
            loss_history = [w['total'] for w in weights_list]
            plt.plot(epochs, loss_history, label='Total Loss', linewidth=2, color='black')
            plt.yscale('log')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (Log Scale)', fontsize=12)
            plt.title(f'Loss Evolution (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()

# 8. Visualization functions
def plot_loss_curves_by_activation(activation_loss_histories: dict, optim_labels: list) -> None:
    for activation_name, loss_histories in activation_loss_histories.items():
        plt.figure(figsize=(10, 6))
        for loss_history, label in zip(loss_histories, optim_labels):
            if not loss_history:  
                continue
                
            # Extract total losses
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
    x_test, y_test = test_data
    n_test = int(np.sqrt(len(x_test)))
    
    # Reshape grids
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
            
            # 1st column: Predicted solution
            ax1 = axes[idx, 0]
            im1 = ax1.pcolormesh(x_grid, y_grid, u_pred_grid, cmap=cm.jet, shading='gouraud')
            ax1.set_xlabel('x', fontsize=10)
            ax1.set_ylabel('y', fontsize=10)
            ax1.set_title(f'{optim_label} - Predicted', fontsize=12)
            plt.colorbar(im1, ax=ax1, shrink=0.6, aspect=5)
            
            # 2nd column: Analytical solution
            ax2 = axes[idx, 1]
            im2 = ax2.pcolormesh(x_grid, y_grid, u_exact_grid, cmap=cm.jet, shading='gouraud')
            ax2.set_title('Analytical Solution', fontsize=12)
            plt.colorbar(im2, ax=ax2, shrink=0.6, aspect=5)
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('y', fontsize=10)
            
            # 3rd column: Absolute Error
            ax3 = axes[idx, 2]
            im3 = ax3.pcolormesh(x_grid, y_grid, error_grid, cmap=cm.viridis, shading='gouraud')
            ax3.set_xlabel('x', fontsize=10)
            ax3.set_ylabel('y', fontsize=10)
            ax3.set_title(f'{optim_label} - Absolute Error', fontsize=12)
            plt.colorbar(im3, ax=ax3, shrink=0.6, aspect=5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

# 9. Main function
def train_2D_poisson_lbPINN_with_activations_and_source_term():
    # 1. Hyperparameter settings
    n_layers = 4
    input_dim = 2
    output_dim = 1
    hidden_dim = 50
    n_pde = 2000      
    n_boundary = 250  
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
    
    # Optimizer labels
    optim_labels = ['Adam (lbPINN)', 'L-BFGS (lbPINN)', 'Adam→L-BFGS (lbPINN)']
    
    # 2. Generate data (including fixed boundary points)
    print("Generating training and testing data...")
    pde_data, boundary_data, test_data = generate_data(
        n_pde=n_pde,
        n_boundary=n_boundary,
        n_test=n_test
    )
    x_test, y_test = test_data
    
    # Calculate exact solution
    model_temp = lbPINN2D(input_dim, output_dim, hidden_dim, 1, TanhActivation())
    u_exact = model_temp.analytical_solution(x_test, y_test)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}  
    activation_weight_histories = {} 
    activation_results = {}
    activation_training_times = {}
    
    # 4. Training loop over activation functions
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name} | Equation parameters: β1={BETA1}, β2={BETA2}")
        print("="*80)
        
        current_loss_histories = []
        current_weight_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Training Adam (lbPINN2D)
        print(f"\n--- {activation_name} + Adam (lbPINN) ---")
        model_adam = lbPINN2D(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
        loss_history_adam = []
        weight_history_adam = []
        start_time_adam = time.time()
        train_lbpinn_with_optimizer(
            model=model_adam,
            optimizer=optimizer_adam,
            epochs=epochs_adam,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam,
            weight_history=weight_history_adam
        )
        training_time_adam = time.time() - start_time_adam
        current_training_times.append(training_time_adam)
        
        # evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam.final_total_loss(pde_data, boundary_data)
        current_results['Adam (lbPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam)
        current_weight_histories.append(weight_history_adam)
        print(f"Adam training time: {training_time_adam:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 Training L-BFGS (lbPINN2D)
        print(f"\n--- {activation_name} + L-BFGS (lbPINN) ---")
        model_lbfgs = lbPINN2D(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_lbfgs = []
        weight_history_lbfgs = []
        start_time_lbfgs = time.time()
        train_lbfgs_lbpinn(
            model=model_lbfgs,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_lbfgs,
            weight_history=weight_history_lbfgs,
            lbfgs_max_iter=max_iter_lbfgs,
            lr_lbfgs=lr_lbfgs_config[activation_name]
        )
        training_time_lbfgs = time.time() - start_time_lbfgs
        current_training_times.append(training_time_lbfgs)
        
        # evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_lbfgs.final_total_loss(pde_data, boundary_data)
        current_results['L-BFGS (lbPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_lbfgs)
        current_weight_histories.append(weight_history_lbfgs)
        print(f"L-BFGS training time: {training_time_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Training Adam→L-BFGS (lbPINN2D)
        print(f"\n--- {activation_name} + Adam→L-BFGS (lbPINN) ---")
        model_adam_lbfgs = lbPINN2D(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_adam_lbfgs = []
        weight_history_adam_lbfgs = []
        start_time_adam_lbfgs = time.time()
        train_adam_lbfgs_lbpinn(
            model=model_adam_lbfgs,
            pde_data=pde_data,
            boundary_data=boundary_data,
            loss_history=loss_history_adam_lbfgs,
            weight_history=weight_history_adam_lbfgs,
            lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        training_time_adam_lbfgs = time.time() - start_time_adam_lbfgs
        current_training_times.append(training_time_adam_lbfgs)
        
        # evaluate
        l2_err, l2_rel_err, linf_err, _, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam_lbfgs.final_total_loss(pde_data, boundary_data)
        current_results['Adam→L-BFGS (lbPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        current_weight_histories.append(weight_history_adam_lbfgs)
        if activation_name == 'Tanh':  
            save_path = "2D_poisson_tanh_adam2lbfgs_lbpinn.pth"
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
            torch.save(complete_save_dict, save_path)
            print(f"✅ Tanh+Adam→LBFGS complete information saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {training_time_adam_lbfgs:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_weight_histories[activation_name] = current_weight_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # 5. Visualization of results
    # 5.1 Plot adaptive weight changes
    plot_adaptive_weights(activation_weight_histories, optim_labels)
    
    # 5.2 Plot loss curves
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    
    # 5.3 Plot solution comparisons
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # 6. Print comprehensive comparison table
    print("\n" + "="*120)
    print("lbPINN 2D Poisson Equation All Activations × Optimizers Comprehensive Comparison Table (Fixed Boundary Points)")
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
    
    # 7. Print best performance per activation function
    print("\n" + "="*100)
    print("lbPINN 2D Poisson Equation Best Performance per Activation Function (Sorted by Standard L2 Error - Fixed Boundary Points)")
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
train_2D_poisson_lbPINN_with_activations_and_source_term()