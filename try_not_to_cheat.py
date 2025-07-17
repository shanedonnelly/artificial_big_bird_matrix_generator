import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def best_diagonal_width(n, s):
    """
    Calculate the best diagonal width for a given matrix size n and sparsity s.
    
    Parameters:
    n (int): matrix length (assuming square matrix)
    s (float): sparsity parameter in [0,1]
    
    Returns:
    int: best diagonal width (odd number)
    """
    # Calculate density
    d = 1 - s
    
    # Calculate target diagonal area
    da = n * n * d
    
    # Quadratic equation coefficients: -x² + (2n-1)x + (n-da) = 0
    a = -1
    b = 2*n - 1
    c = n - da
    
    # Calculate discriminant
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return 1  # Return minimum odd diagonal width
    
    # Calculate the positive solution
    x = (-b + math.sqrt(discriminant)) / (2*a)
    
    # Round to nearest integer to get sdw
    sdw = round(x)
    
    # Ensure sdw is non-negative
    sdw = max(0, sdw)
    
    # Calculate diagonal width (must be odd)
    diagonal_width = 2 * sdw + 1
    
    # Ensure it's within bounds [1, 2n-1] and odd
    diagonal_width = max(1, min(diagonal_width, 2*n-1))
    if diagonal_width % 2 == 0:
        diagonal_width += 1
    
    return diagonal_width

def calculate_actual_sparsity(n, diagonal_width):
    """
    Calculate the actual sparsity for a given diagonal width.
    """
    if diagonal_width == 0:
        return 1.0
    
    sdw = (diagonal_width - 1) // 2
    diagonal_area = n * (1 + 2*sdw) - sdw * (sdw + 1)
    density = diagonal_area / (n * n)
    sparsity = 1 - density
    
    return sparsity

# Pre-calculate all results for different n values
n_values = np.arange(10, 1001, 10)  # From 10 to 1000, step 10
sparsity_values = np.arange(0.0, 1.0, 0.05)  # 20 values from 0 to 0.95

# Pre-calculate results for all combinations
print("Pre-calculating results...")
all_results = {}
for n in n_values:
    results = []
    for s in sparsity_values:
        diagonal_width = best_diagonal_width(n, s)
        actual_sparsity = calculate_actual_sparsity(n, diagonal_width)
        
        results.append({
            'target_sparsity': s,
            'diagonal_width': diagonal_width,
            'actual_sparsity': actual_sparsity,
            'error': abs(s - actual_sparsity)
        })
    
    all_results[n] = results

print("Pre-calculation complete!")

# Create interactive plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.25)

# Initial n value
initial_n = 100
initial_results = all_results[initial_n]

target_sparsities = [r['target_sparsity'] for r in initial_results]
actual_sparsities = [r['actual_sparsity'] for r in initial_results]
errors = [r['error'] for r in initial_results]

# Plot 1: Actual vs target sparsity
line1_actual, = ax1.plot(target_sparsities, actual_sparsities, 'r-', linewidth=2, label='Actual')
line1_target, = ax1.plot(target_sparsities, target_sparsities, 'k--', linewidth=1, label='Target')
ax1.set_xlabel('Target Sparsity')
ax1.set_ylabel('Sparsity')
ax1.set_title(f'Actual vs Target Sparsity (n={initial_n})')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Plot 2: Error
line2, = ax2.plot(target_sparsities, errors, 'g-', linewidth=2)
ax2.set_xlabel('Target Sparsity')
ax2.set_ylabel('Absolute Error')
ax2.set_title(f'Error in Sparsity Approximation (n={initial_n})')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

# Add slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Matrix Size (n)', 10, 1000, valinit=initial_n, valstep=10)

# Update function
def update(val):
    n = int(slider.val)
    results = all_results[n]
    
    target_sparsities = [r['target_sparsity'] for r in results]
    actual_sparsities = [r['actual_sparsity'] for r in results]
    errors = [r['error'] for r in results]
    
    # Update plots
    line1_actual.set_ydata(actual_sparsities)
    line2.set_ydata(errors)
    
    # Update titles
    ax1.set_title(f'Actual vs Target Sparsity (n={n})')
    ax2.set_title(f'Error in Sparsity Approximation (n={n})')
    
    # Update y-axis limits for error plot
    ax2.set_ylim(0, max(errors) * 1.1)
    
    # Print statistics
    print(f"\nStatistics for n = {n}:")
    print(f"Maximum error: {max(errors):.4f}")
    print(f"Average error: {np.mean(errors):.4f}")
    print(f"Median error: {np.median(errors):.4f}")
    
    fig.canvas.draw()

slider.on_changed(update)

# Initial statistics
print(f"\nStatistics for n = {initial_n}:")
print(f"Maximum error: {max(errors):.4f}")
print(f"Average error: {np.mean(errors):.4f}")
print(f"Median error: {np.median(errors):.4f}")

plt.tight_layout()
plt.show()