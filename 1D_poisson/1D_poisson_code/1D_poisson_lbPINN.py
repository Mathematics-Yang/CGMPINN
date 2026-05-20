import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# define constants for the source term
ALPHA1 = 5.0  # sine term parameter
ALPHA2 = 3.0  # cosine term parameter
k = 20.0      # steepness of the hyperbolic tangent function

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
        
        # initialize adaptive weight parameters
        self.log_var_pde = nn.Parameter(torch.tensor(0.0))    # log variance for PDE loss
        self.log_var_boundary = nn.Parameter(torch.tensor(0.0))# log variance for boundary loss

        # regularization coefficient
        self.reg_coeff = 0.5

    # forward propagation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        u = self.forward(x)
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
        return u_xx

    # define source term f(x)
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        pi = np.pi
        alpha1pi = ALPHA1 * pi
        alpha2pi = ALPHA2 * pi
        term1_second = -(alpha1pi**2 + alpha2pi**2) * torch.sin(alpha1pi * x) * torch.cos(alpha2pi * x) \
               - 2 * alpha1pi * alpha2pi * torch.cos(alpha1pi * x) * torch.sin(alpha2pi * x)
        sech_kx = 1 / torch.cosh(k * x)
        term2_second = -2 * k**2 * sech_kx**2 * torch.tanh(k * x)
        f = term1_second + term2_second
        return f

    def pde_residual(self, x: torch.Tensor) -> torch.Tensor:
        u_xx_pred = self.compute_second_derivative(x)
        f_exact = self.source_term(x)
        return u_xx_pred - f_exact
    
    def boundary_residual(self) -> torch.Tensor:
        x0 = torch.tensor([[0.0]], requires_grad=True)
        u0_pred = self.forward(x0)
        u0_exact = self.analytical_solution(x0)
        x1 = torch.tensor([[1.0]], requires_grad=True)
        u1_pred = self.forward(x1)
        u1_exact = self.analytical_solution(x1)
        return torch.cat([u0_pred - u0_exact, u1_pred - u1_exact], dim=0)
    
    # analytical solution
    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(ALPHA1 * np.pi * x) * torch.cos(ALPHA2 * np.pi * x) + torch.tanh(k * x)
    
    # adaptive loss function
    def adaptive_loss(self, pde_data) -> torch.Tensor:
        """
        Adaptive loss function (Gaussian likelihood + regularization)
        Poisson equation only includes PDE loss and boundary loss
        """
        x_pde = pde_data
        pde_res = self.pde_residual(x_pde)
        boundary_res = self.boundary_residual()
        
        # adaptive weights (log variance parameterization)
        raw_weight_pde = 0.5 * torch.exp(-self.log_var_pde)
        raw_weight_boundary = 0.5 * torch.exp(-self.log_var_boundary)

        # weighted losses
        loss_pde = raw_weight_pde * torch.mean(pde_res ** 2)
        loss_boundary = raw_weight_boundary * torch.mean(boundary_res ** 2)
        
        # regularization term (to prevent extreme weights)
        reg_term = self.reg_coeff * (F.softplus(self.log_var_pde) + F.softplus(self.log_var_boundary))
        
        # total loss
        total_loss = loss_pde + loss_boundary + reg_term
        return total_loss, loss_pde, loss_boundary
    
    # compute final total loss (for evaluation)
    def final_total_loss(self, pde_data) -> tuple[float, float, float]:
        x_pde = pde_data
        
        with torch.enable_grad():
            total_loss, pde_loss, boundary_loss = self.adaptive_loss(pde_data)
            return (total_loss.item(), pde_loss.item(), boundary_loss.item())
    
    # get adaptive weights
    def get_adaptive_weights(self) -> dict:
        # raw weights
        raw_weight_pde = 0.5 * torch.exp(-self.log_var_pde).item()
        raw_weight_boundary = 0.5 * torch.exp(-self.log_var_boundary).item()
        
        # log variances
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

# 2. generate training and testing data
def generate_data(
    n_pde: int,
    n_test: int 
) -> tuple:
    x_pde = torch.rand(n_pde, 1, requires_grad=True)
    pde_data = x_pde
    x_test = torch.linspace(0.0, 1.0, n_test).reshape(-1, 1)
    test_data = x_test
    
    return pde_data, test_data

# 3. Adam training function
def train_lbpinn_with_optimizer(
    model: lbPINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: torch.Tensor,
    loss_history: list,
    weight_history: list
) -> None:
    model.train()
    for epoch in range(epochs):
        total_loss, pde_loss, boundary_loss = model.adaptive_loss(pde_data)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # record losses
        loss_history.append({
            'total': total_loss.item(),
            'pde': pde_loss.item(),
            'boundary': boundary_loss.item()
        })
        
        # record weights every 5 epochs
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
    model: lbPINN,
    pde_data: torch.Tensor,
    loss_history: list,
    weight_history: list,  
    lbfgs_max_iter: int,   
    lr_lbfgs: float      
) -> None:
    model.train()
    
    # store component loss values
    pde_loss_val = boundary_loss_val = 0.0

    # closure function for L-BFGS
    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, boundary_loss_val
        
        optimizer.zero_grad()
        total_loss, pde_loss, boundary_loss = model.adaptive_loss(pde_data)
        total_loss.backward()
        
        # store component loss values
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
        
        # record weights
        if iter_idx % 5 == 0:
            weights = model.get_adaptive_weights()
            weights['total'] = current_loss
            weight_history.append(weights)
        
        # print information
        if (iter_idx + 1) % 500 == 0 or iter_idx == 0:
            weights = model.get_adaptive_weights()
            print(
                f'Iteration {iter_idx+1}/{lbfgs_max_iter} | Total Loss: {current_loss:.2e} | '
                f'PDE Loss: {pde_loss_val:.2e} | Boundary Loss: {boundary_loss_val:.2e} | '
                f'Weights: [PDE: {weights["raw_weight_pde"]:.2e}, Boundary: {weights["raw_weight_boundary"]:.2e}]'
            )

# 5. Two-stage training function (Adam→L-BFGS)
def train_adam_lbfgs_lbpinn(
    model: lbPINN,
    pde_data: torch.Tensor,
    loss_history: list,
    weight_history: list,
    adam_epochs: int,
    lr_lbfgs: float,
    lbfgs_max_iter: int
) -> None:
    print("=== Stage 1: Adam Global Exploration ===")
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    train_lbpinn_with_optimizer(
        model=model,
        optimizer=optimizer_adam,
        epochs=adam_epochs,
        pde_data=pde_data,
        loss_history=loss_history,
        weight_history=weight_history
    )
    
    print("\n=== Stage 2: L-BFGS Local Refinement ===")
    train_lbfgs_lbpinn(
        model=model,
        pde_data=pde_data,
        loss_history=loss_history,
        weight_history=weight_history,
        lbfgs_max_iter=lbfgs_max_iter,
        lr_lbfgs=lr_lbfgs
    )

# 6. Evaluation function
def evaluate_model(
    model: lbPINN,
    test_data: torch.Tensor,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    x_test = test_data
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test)
        
        # L2 error
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        
        # L2 relative error
        u_exact_l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()
        l2_relative_error = l2_error / u_exact_l2_norm if u_exact_l2_norm > 1e-10 else float('inf')
        
        # L∞ error
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        
        # Pointwise error and predictions
        pointwise_error = torch.abs(u_pred - u_exact).numpy()
        u_pred_np = u_pred.numpy()
    
    return l2_error, l2_relative_error, linf_error, pointwise_error, u_pred_np

# 7. Visualization function
def plot_adaptive_weights(weight_histories: dict, optim_labels: list) -> None:
    for activation_name, optim_weights_list in weight_histories.items():
        for opt_idx, (weights_list, opt_label) in enumerate(zip(optim_weights_list, optim_labels)):
            if not weights_list:
                continue
                
            plt.figure(figsize=(10, 10))
            epochs = [i*5 for i in range(len(weights_list))]
            
            # Extract data
            raw_weight_pde = [w['raw_weight_pde'] for w in weights_list]
            raw_weight_boundary = [w['raw_weight_boundary'] for w in weights_list]
            raw_log_var_pde = [w['raw_log_var_pde'] for w in weights_list]
            raw_log_var_boundary = [w['raw_log_var_boundary'] for w in weights_list]
            
            # 1. Raw Weights
            plt.subplot(3, 1, 1)
            plt.plot(epochs, raw_weight_pde, label='PDE Weight', linewidth=2)
            plt.plot(epochs, raw_weight_boundary, label='Boundary Weight', linewidth=2)
            plt.yscale('log')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Raw Weight (Log Scale)', fontsize=12)
            plt.title(f'Adaptive Weights (Activation: {activation_name}, Optimizer: {opt_label})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
        
            # 2. Log-Variance
            plt.subplot(3, 1, 2)
            plt.plot(epochs, raw_log_var_pde, label='PDE Log-Var', linewidth=2)
            plt.plot(epochs, raw_log_var_boundary, label='Boundary Log-Var', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Log-Variance', fontsize=12)
            plt.legend()
            
            # 3. Total Loss
            plt.subplot(3, 1, 3)
            loss_history = [w['total'] for w in weights_list]
            plt.plot(epochs, loss_history, label='Total Loss', linewidth=2, color='black')
            plt.yscale('log')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss (Log Scale)', fontsize=12)
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
    test_data: torch.Tensor,
    u_exact_np: np.ndarray,
    optim_labels: list
) -> None:
    x_test_np = test_data.numpy()
    
    for activation_name, results in activation_results.items():
        fig, axes = plt.subplots(len(optim_labels), 2, figsize=(14, 4*len(optim_labels)))
        fig.suptitle(f'Solution Comparison - Activation: {activation_name}', fontsize=18, y=0.98)
        
        for idx, optim_label in enumerate(optim_labels):
            u_pred = results[optim_label]['u_pred']
            pointwise_error = results[optim_label]['pointwise_error']
            
            # Solution Comparison
            ax1 = axes[idx, 0] if len(optim_labels) > 1 else axes[0]
            ax1.plot(x_test_np, u_exact_np, 'b-', label='Analytical', linewidth=2)
            ax1.plot(x_test_np, u_pred, 'r--', label='lbPINN Prediction', linewidth=2, alpha=0.8)
            ax1.set_xlabel('x', fontsize=10)
            ax1.set_ylabel('u(x)', fontsize=10)
            ax1.set_title(f'{optim_label} - Solution', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Pointwise Error
            ax2 = axes[idx, 1] if len(optim_labels) > 1 else axes[1]
            ax2.plot(x_test_np, pointwise_error, 'g-', linewidth=2)
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('Absolute Error', fontsize=10)
            ax2.set_title(f'{optim_label} - Pointwise Error', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

# 8. main training function
def train_1D_poisson_lbPINN_with_activations_and_source_term():
    # Hyperparameter settings
    n_layers = 4
    input_dim = 1  
    output_dim = 1
    hidden_dim = 50
    n_pde = 1500
    n_test = 200
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
    
    # Generate data
    print("Generating training and testing data...")
    pde_data, test_data = generate_data(n_pde=n_pde, n_test=n_test)
    
    # Compute analytical solution
    model_temp = lbPINN(input_dim, output_dim, hidden_dim, 1, TanhActivation())
    u_exact = model_temp.analytical_solution(test_data)
    u_exact_np = u_exact.numpy()
    
    # Initialize storage variables
    activation_loss_histories = {}
    activation_weight_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # Train by activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation function: {activation_name}")
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
            loss_history=loss_history_adam,
            weight_history=weight_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam.final_total_loss(pde_data)
        current_results['Adam (lbPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred,
            'pointwise_error': pointwise_err
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
            loss_history=loss_history_lbfgs,
            weight_history=weight_history_lbfgs,
            lbfgs_max_iter=max_iter_lbfgs,
            lr_lbfgs=lr_lbfgs_config[activation_name]
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_lbfgs.final_total_loss(pde_data)
        current_results['L-BFGS (lbPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred,
            'pointwise_error': pointwise_err
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
            loss_history=loss_history_adam_lbfgs,
            weight_history=weight_history_adam_lbfgs,
            lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluation
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam_lbfgs.final_total_loss(pde_data)
        current_results['Adam→L-BFGS (lbPINN)'] = {
            'l2_error': l2_err,
            'l2_relative_error': l2_rel_err,
            'linf_error': linf_err,
            'final_total_loss': final_total,
            'final_pde_loss': final_pde,
            'final_boundary_loss': final_boundary,
            'u_pred': u_pred,
            'pointwise_error': pointwise_err
        }
        current_loss_histories.append(loss_history_adam_lbfgs)
        current_weight_histories.append(weight_history_adam_lbfgs)
        if activation_name == 'Tanh':  
            # Define save path
            save_path = "1D_poisson_tanh_adam2lbfgs_lbpinn.pth"
    
            # Pack all contents to be saved
            complete_save_dict = {
                'model_state_dict': model_adam_lbfgs.state_dict(),  # Model learnable parameters (core)
                'loss_history': loss_history_adam_lbfgs,            # Loss history
                'weight_history': weight_history_adam_lbfgs,        # Adaptive weight history
                'final_total_loss': final_total,                    # Final total loss
                'training_time': train_time,                        # Training runtime
                'activation_name': activation_name,                 # Additional auxiliary information 
                'hyper_parameters': {                               # Hyperparameters 
                    'n_layers': n_layers,
                    'hidden_dim': hidden_dim,
                    'adam_epochs': adam_lbfgs_adam_epochs,
                    'lbfgs_max_iter': adam_lbfgs_lbfgs_iter
                }
            }
    
            # Save to file
            torch.save(complete_save_dict, save_path)
            print(f"✅ Tanh+Adam→LBFGS complete information saved to: {save_path}")
        print(f"Adam→L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # Save results
        activation_loss_histories[activation_name] = current_loss_histories
        activation_weight_histories[activation_name] = current_weight_histories
        activation_results[activation_name] = current_results
        activation_training_times[activation_name] = current_training_times
    
    # Visualization
    plot_adaptive_weights(activation_weight_histories, optim_labels)
    plot_loss_curves_by_activation(activation_loss_histories, optim_labels)
    plot_solutions_by_activation_and_optimizer(activation_results, test_data, u_exact_np, optim_labels)
    
    # Print comparison table
    print("\n" + "="*120)
    print("lbPINN 1D Poisson Equation Comprehensive Comparison Table")
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
    
    # Detailed ranking by activation function
    print("\n" + "="*100)
    print("lbPINN Best Performance by Activation Function (Sorted by L2 Error)")
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
        # Sort and print
        optim_results_sorted = sorted(optim_results, key=lambda x: x['l2_error'])
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<20} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_poisson_lbPINN_with_activations_and_source_term()