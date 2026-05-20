import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# set random seeds for reproducibility
torch.manual_seed(1224)
np.random.seed(1224)

# define activation functions
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# define equation parameters
ALPHA1 = 5.0  # sine term parameter
ALPHA2 = 3.0  # cosine term parameter
k = 20.0      # steepness of the hyperbolic tangent function

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

    # define source term f(x) = u_xx from analytical solution
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        pi = np.pi
        alpha1pi = ALPHA1 * pi
        alpha2pi = ALPHA2 * pi
        
        # second derivative of the first term: d²/dx² [sin(α1πx)cos(α2πx)]
        term1_second = -(alpha1pi**2 + alpha2pi**2) * torch.sin(alpha1pi * x) * torch.cos(alpha2pi * x) \
               - 2 * alpha1pi * alpha2pi * torch.cos(alpha1pi * x) * torch.sin(alpha2pi * x)
        
        # second derivative of tanh(kx)
        sech_kx = 1 / torch.cosh(k * x)
        term2_second = -2 * k**2 * sech_kx**2 * torch.tanh(k * x)
        
        # combine both terms
        u_xx_exact = term1_second + term2_second
        return u_xx_exact

    def pde_loss(self, x: torch.Tensor) -> torch.Tensor:
        u_xx_pred = self.compute_second_derivative(x)
        f_exact = self.source_term(x)
        pde_residual = u_xx_pred - f_exact
        return torch.mean(pde_residual ** 2)
    
    def boundary_loss(self) -> torch.Tensor:
        x0 = torch.tensor([[0.0]], requires_grad=True)
        u0_pred = self.forward(x0)
        u0_exact = self.analytical_solution(x0)
        x1 = torch.tensor([[1.0]], requires_grad=True)
        u1_pred = self.forward(x1)
        u1_exact = self.analytical_solution(x1)
        boundary_residual = (u0_pred - u0_exact)**2 + (u1_pred - u1_exact)**2
        return torch.mean(boundary_residual)
    
    # analytical solution
    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        """Analytical solution: u(x) = sin(α1πx)cos(α2πx) + tanh(kx)"""
        return torch.sin(ALPHA1 * np.pi * x) * torch.cos(ALPHA2 * np.pi * x) + torch.tanh(k * x)
    
    # compute final total loss
    def final_total_loss(self, pde_data) -> tuple[float, float, float]:
        x_pde = pde_data
        
        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde).item()
            boundary_loss_val = self.boundary_loss().item()
            total_loss_val = pde_loss_val + boundary_loss_val
        
        return total_loss_val, pde_loss_val, boundary_loss_val

# 2. Data generation function
def generate_data(
    n_pde: int,
    n_test: int 
) -> tuple:
    """
    Generate training data (PDE interior points) and test data
    Returns: (pde_data, test_data)
    """
    # PDE interior points: x ∈ [0,1]
    x_pde = torch.rand(n_pde, 1, requires_grad=True)
    pde_data = x_pde
    
    # Test data (uniform grid)
    x_test = torch.linspace(0.0, 1.0, n_test).reshape(-1, 1)
    test_data = x_test
    
    return pde_data, test_data

# 3. Training function (Adam)
def train_with_optimizer(
    model: PINN,
    optimizer: optim.Optimizer,
    epochs: int,
    pde_data: torch.Tensor,
    loss_history: list
) -> None:
    model.train()

    for epoch in range(epochs):
        pde_loss = model.pde_loss(pde_data)
        boundary_loss = model.boundary_loss()
        total_loss = pde_loss + boundary_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if (epoch + 1) % 1000 == 0:
            print(
                f'Epoch {epoch+1:4d} | Total Loss: {total_loss.item():.2e} | '
                f'PDE Loss: {pde_loss.item():.2e} | Boundary Loss: {boundary_loss.item():.2e}'
            )

# 4. Training function (L-BFGS)
def train_with_lbfgs(
    model: PINN,
    pde_data: torch.Tensor,
    loss_history: list,
    lr: float,
    max_iter: int
) -> None:
    model.train()
    pde_loss_val = boundary_loss_val = 0.0

    def closure() -> torch.Tensor:
        nonlocal pde_loss_val, boundary_loss_val
        optimizer.zero_grad()
        pde_loss = model.pde_loss(pde_data)
        boundary_loss = model.boundary_loss()
        total_loss = pde_loss + boundary_loss
        total_loss.backward()
        
        pde_loss_val = pde_loss.item()
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
                f'PDE Loss: {pde_loss_val:.2e} | Boundary Loss: {boundary_loss_val:.2e}'
            )

# 5. Two-stage training function (Adam→L-BFGS)
def train_adam_lbfgs(
    model: PINN,
    pde_data: torch.Tensor,
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
        loss_history=loss_history
    )
    
    print("\n=== Stage 2: L-BFGS Local Refinement ===")
    train_with_lbfgs(
        model=model,
        pde_data=pde_data,
        loss_history=loss_history,
        lr=lr_lbfgs,
        max_iter=lbfgs_max_iter
    )

# 6. Evaluation function
def evaluate_model(
    model: PINN,
    test_data: torch.Tensor,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model: compute standard L2 error, relative L2 error, and L∞ error
    Returns: (l2_error, l2_relative_error, linf_error, pointwise_error, u_pred_np)
    """
    x_test = test_data
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test)
        
        # Standard L2 error
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        
        # Relative L2 error
        u_exact_l2_norm = torch.sqrt(torch.mean(u_exact ** 2)).item()
        l2_relative_error = l2_error / u_exact_l2_norm if u_exact_l2_norm > 1e-10 else float('inf')
        
        # L∞ error (maximum absolute error)
        linf_error = torch.max(torch.abs(u_pred - u_exact)).item()
        
        # Pointwise error and predictions
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

            # solution comparison
            ax1 = axes[idx, 0] if len(optim_labels) > 1 else axes[0]
            ax1.plot(x_test_np, u_exact_np, 'b-', label='Analytical Solution', linewidth=2)
            ax1.plot(x_test_np, u_pred, 'r--', label='PINN Prediction', linewidth=2, alpha=0.8)
            ax1.set_xlabel('x', fontsize=10)
            ax1.set_ylabel('u(x)', fontsize=10)
            ax1.set_title(f'{optim_label} - Solution', fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # pointwise error
            ax2 = axes[idx, 1] if len(optim_labels) > 1 else axes[1]
            ax2.plot(x_test_np, pointwise_error, 'g-', linewidth=2)
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('Absolute Error', fontsize=10)
            ax2.set_title(f'{optim_label} - Pointwise Error', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

# 8. Main function (execution flow)
def train_1D_poisson_PINN_with_activations_and_source_term():
    # 1. Hyperparameter settings
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
    
    # define activation functions to test
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (PINN)', 'L-BFGS (PINN)', 'Adam→L-BFGS (PINN)']
    
    # 2. Generate data (shared across all activation functions)
    print("Generating training and testing data...")
    pde_data, test_data = generate_data(
        n_pde=n_pde,
        n_test=n_test
    )
    
    # Calculate exact solution
    model_temp = PINN(input_dim, output_dim, hidden_dim, 1, TanhActivation())  
    u_exact = model_temp.analytical_solution(test_data)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation Function: {activation_name} | Equation Parameters: α1={ALPHA1}, α2={ALPHA2}, k={k}")
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
            loss_history=loss_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam.final_total_loss(pde_data)
        current_results['Adam (PINN)'] = {
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
        print(f"Adam training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.2 L-BFGS training
        print(f"\n--- {activation_name} + L-BFGS ---")
        model_lbfgs = PINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_lbfgs = []
        start_time = time.time()
        train_with_lbfgs(
            model=model_lbfgs,
            pde_data=pde_data,
            loss_history=loss_history_lbfgs,
            lr=lr_lbfgs_config[activation_name],
            max_iter=max_iter_lbfgs
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate L-BFGS
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_lbfgs.final_total_loss(pde_data)
        current_results['L-BFGS (PINN)'] = {
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
        print(f"L-BFGS training time: {train_time:.2f}s | L2 error: {l2_err:.2e} | L∞ error: {linf_err:.2e}")
        
        # 4.3 Adam→L-BFGS training
        print(f"\n--- {activation_name} + Adam→L-BFGS ---")
        model_adam_lbfgs = PINN(input_dim, output_dim, hidden_dim, n_layers, activation_fn)
        loss_history_adam_lbfgs = []
        start_time = time.time()
        train_adam_lbfgs(
            model=model_adam_lbfgs,
            pde_data=pde_data,
            loss_history=loss_history_adam_lbfgs,
            lr_lbfgs=lr_adam_lbfgs,
            adam_epochs=adam_lbfgs_adam_epochs,
            lbfgs_max_iter=adam_lbfgs_lbfgs_iter
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam→L-BFGS
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam_lbfgs, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam_lbfgs.final_total_loss(pde_data)
        current_results['Adam→L-BFGS (PINN)'] = {
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
        if activation_name == 'Tanh':  
            # Define save path
            save_path = "1D_poisson_tanh_adam2lbfgs_pinn.pth"
    
            # Pack all contents to be saved
            complete_save_dict = {
                'model_state_dict': model_adam_lbfgs.state_dict(),  # Model learnable parameters (core)
                'loss_history': loss_history_adam_lbfgs,            # Loss history
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
    
            # Save custom dictionary
            torch.save(complete_save_dict, save_path)
            print(f"✅ Tanh+Adam→LBFGS complete information saved to: {save_path}")
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
    print("Comprehensive comparison table for all activation functions × optimizers (1D Poisson equation)")
    print("="*120)
    header = (f"{'Activation':<10} {'Optimizer':<15} {'Training Time(s)':<12} {'Standard L2 Error':<12} "
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
        # sort by l2_error
        optim_results_sorted = sorted(optim_results, key=lambda x: x['l2_error'])
        for i, res in enumerate(optim_results_sorted, 1):
            print(
                f"  Rank {i}: {res['optimizer']:<12} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute the main function
train_1D_poisson_PINN_with_activations_and_source_term()