import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

torch.manual_seed(1224)
np.random.seed(1224)

# Define activation function
class TanhActivation(nn.Module):
    def forward(self, x):
        return torch.tanh(x)

# Define equation parameters
ALPHA1 = 5.0  
ALPHA2 = 3.0  
k = 20.0      

# Define lightweight PINN block (basic module of STAR-PINN)
class LightweightPINNBlock(nn.Module):
    """Lightweight PINN block for stacking in STAR-PINN"""
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation):
        super(LightweightPINNBlock, self).__init__()
        self.activation = activation
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

# 1. STAR-PINN model
class STAR_PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers_per_block, activation, n_blocks=3):
        super(STAR_PINN, self).__init__()
        self.n_blocks = n_blocks  # number of stacked lightweight PINN blocks
        self.activation = activation
        
        # Create stacked lightweight PINN blocks
        self.pinn_blocks = nn.ModuleList([
            LightweightPINNBlock(input_dim, output_dim, hidden_dim, n_layers_per_block, activation)
            for _ in range(n_blocks)
        ])
        
        # Adaptive residual weights (trainable)
        self.adaptive_weights = nn.Parameter(torch.ones(n_blocks - 1) * 0.5)  # α1, α2,...α(n_blocks-1)
    
    # Forward propagation: stacked residual fusion
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation: stacked adaptive residual fusion
        A_z^0 = PINN0(x)
        A_z^1 = PINN1(x) + α1*A_z^0
        A_z^2 = PINN2(x) + α2*A_z^1
        ...
        Final output = fusion result of the last block
        """
        # Output of the first block
        current_output = self.pinn_blocks[0](x)
        
        # Subsequent blocks: residual fusion
        for i in range(1, self.n_blocks):
            block_output = self.pinn_blocks[i](x)
            # Adaptive weight fusion (weights constrained in [0,1] range)
            alpha = torch.sigmoid(self.adaptive_weights[i-1])  # Ensure weights are non-negative and reasonable
            current_output = block_output + alpha * current_output
        
        return current_output

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

    # Define source term f(x) = u_xx 
    def source_term(self, x: torch.Tensor) -> torch.Tensor:
        pi = np.pi
        alpha1pi = ALPHA1 * pi
        alpha2pi = ALPHA2 * pi
        term1_second = -(alpha1pi**2 + alpha2pi**2) * torch.sin(alpha1pi * x) * torch.cos(alpha2pi * x) \
               - 2 * alpha1pi * alpha2pi * torch.cos(alpha1pi * x) * torch.sin(alpha2pi * x)
        sech_kx = 1 / torch.cosh(k * x)
        term2_second = -2 * k**2 * sech_kx**2 * torch.tanh(k * x)
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
    
    def analytical_solution(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(ALPHA1 * np.pi * x) * torch.cos(ALPHA2 * np.pi * x) + torch.tanh(k * x)
    
    def final_total_loss(self, pde_data) -> tuple[float, float, float]:
        x_pde = pde_data
        
        with torch.enable_grad():
            pde_loss_val = self.pde_loss(x_pde).item()
            boundary_loss_val = self.boundary_loss().item()
            total_loss_val = pde_loss_val + boundary_loss_val
        
        return total_loss_val, pde_loss_val, boundary_loss_val

# 2. Generate training and testing data
def generate_data(
    n_pde: int,
    n_test: int 
) -> tuple:
    x_pde = torch.rand(n_pde, 1, requires_grad=True)
    pde_data = x_pde
    x_test = torch.linspace(0.0, 1.0, n_test).reshape(-1, 1)
    test_data = x_test
    
    return pde_data, test_data

# 3. Training function (Adam)
def train_with_optimizer(
    model: STAR_PINN,
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
    model: STAR_PINN,
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
    model: STAR_PINN,
    pde_data: torch.Tensor,
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

# 6. Evaluation function (with multiple error metrics)
def evaluate_model(
    model: STAR_PINN,
    test_data: torch.Tensor,
    u_exact: torch.Tensor
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    x_test = test_data
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test)
        
        # Standard L2 error
        l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
        
        # L2 relative error
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
            
            # Solution comparison
            ax1 = axes[idx, 0] if len(optim_labels) > 1 else axes[0]
            ax1.plot(x_test_np, u_exact_np, 'b-', label='Analytical Solution', linewidth=2)
            ax1.plot(x_test_np, u_pred, 'r--', label='STAR-PINN Prediction', linewidth=2, alpha=0.8)
            ax1.set_xlabel('x', fontsize=10)
            ax1.set_ylabel('u(x)', fontsize=10)
            ax1.set_title(f'{optim_label} - Solution', fontsize=12)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Pointwise error
            ax2 = axes[idx, 1] if len(optim_labels) > 1 else axes[1]
            ax2.plot(x_test_np, pointwise_error, 'g-', linewidth=2)
            ax2.set_xlabel('x', fontsize=10)
            ax2.set_ylabel('Absolute Error', fontsize=10)
            ax2.set_title(f'{optim_label} - Pointwise Error', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.show()

# 8. main training routine
def train_1D_poisson_STAR_PINN_with_activations_and_source_term():
    # 1. Hyperparameter settings
    n_layers_per_block = 2    # Number of hidden layers per lightweight PINN block 
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
    
    activations = {'Tanh': TanhActivation()}
    activation_names = list(activations.keys())
    optim_labels = ['Adam (STAR-PINN)', 'L-BFGS (STAR-PINN)', 'Adam→L-BFGS (STAR-PINN)']
    
    # 2. generate data
    print("Generating training and testing data...")
    pde_data, test_data = generate_data(
        n_pde=n_pde,
        n_test=n_test
    )
    
    # Calculate exact solution
    model_temp = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, TanhActivation())
    u_exact = model_temp.analytical_solution(test_data)
    u_exact_np = u_exact.numpy()
    
    # 3. Initialize storage variables
    activation_loss_histories = {}
    activation_results = {}
    activation_training_times = {}
    
    # 4. Train all optimizers for each activation function
    for activation_name, activation_fn in activations.items():
        print("\n" + "="*80)
        print(f"Starting training - Activation: {activation_name} | Equation parameters: α1={ALPHA1}, α2={ALPHA2}, k={k}")
        print(f"STAR-PINN configuration: {3} stacked blocks × {2} hidden layers per block")
        print("="*80)
        
        current_loss_histories = []
        current_results = {}
        current_training_times = []
        
        # 4.1 Adam training (STAR-PINN)
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
            loss_history=loss_history_adam
        )
        train_time = time.time() - start_time
        current_training_times.append(train_time)
        
        # Evaluate Adam
        l2_err, l2_rel_err, linf_err, pointwise_err, u_pred = evaluate_model(model_adam, test_data, u_exact)
        final_total, final_pde, final_boundary = model_adam.final_total_loss(pde_data)
        current_results['Adam (STAR-PINN)'] = {
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
        model_lbfgs = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, activation_fn)
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
        current_results['L-BFGS (STAR-PINN)'] = {
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
        model_adam_lbfgs = STAR_PINN(input_dim, output_dim, hidden_dim, n_layers_per_block, activation_fn)
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
        current_results['Adam→L-BFGS (STAR-PINN)'] = {
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
            save_path = "1D_poisson_tanh_adam2lbfgs_star_pinn.pth"
    
            # Pack all contents to be saved
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
    print("Comprehensive Comparison Table for All Activation Functions × Optimizers (1D Poisson Equation - STAR-PINN)")
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
    
    # 7. Print best performance for each activation function
    print("\n" + "="*100)
    print("Best Performance for Each Activation Function (Sorted by Standard L2 Error - STAR-PINN)")
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
                f"  Rank {i}: {res['optimizer']:<20} | Training Time: {res['train_time']:.2f}s | "
                f"L2 Error: {res['l2_error']:.2e} | L∞ Error: {res['linf_error']:.2e} | "
                f"Final Loss: {res['final_loss']:.2e}"
            )

# Execute main function
train_1D_poisson_STAR_PINN_with_activations_and_source_term()