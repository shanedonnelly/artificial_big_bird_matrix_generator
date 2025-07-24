import numpy as np
import matplotlib.pyplot as plt
import math

## ALL REQUIRED CODE FROM NOTEBOOK

def get_nb_non_zero(matrix):
    return np.count_nonzero(matrix)

def get_density(matrix, size):
    return float(get_nb_non_zero(matrix)) / float(size * size)

def get_sparsity(matrix, size):
    return 1.0 - get_density(matrix, size)

def get_random_attention_mask(size, nz_per_row):
    # conditions : nz_per_row <= size
    rng = np.random.default_rng(121263137472525314065)
    mask = rng.multivariate_hypergeometric([1]*size, nz_per_row, size=size).astype(bool)
    return mask

def best_nz_per_row_from_sparsity(size, sparsity):
    # conditions : 0 <= sparsity <= 1
    return round(size * (1 - sparsity))

def get_random_attention_mask_with_sparsity(size, sparsity):
    # conditions : 0 <= sparsity <= 1
    nz_per_row=best_nz_per_row_from_sparsity(size=size, sparsity=sparsity)
    return get_random_attention_mask( size=size, nz_per_row=nz_per_row)

def diagonal_area(size,diagonal_width):
    # conditions : size
    if(diagonal_width == 0):
        return 0
    else:
        n = size
        #semi diagonal widht
        sdw = diagonal_width // 2 # (diagonal_width / 2 - 1 because is odd)
        da = n * ( 1 + 2 * sdw ) - sdw * (sdw + 1)
        return da

def get_window_attention_mask (size, diagonal_width):
    # conditions : shape(matrix) = (size, size), 0 <= diagonal_width <= 2*size - 1 (cover full matrix), diagonal_width is odd
    mask = np.zeros(shape=(size, size), dtype=bool)
    if (diagonal_width > 0):
        sdw = diagonal_width // 2
        if diagonal_width == 1:
            mask = np.fromfunction(lambda i, j:  j == i,shape=(size, size), dtype=int)
        else : 
            mask = np.fromfunction(lambda i, j:  np.abs(i - j) <= sdw ,shape=(size, size), dtype=int)
    return mask

def best_diagonal_width_from_sparsity(size, sparsity):
    n = size
    density = 1.0 - sparsity
     # ideal diagonal aera
    da = n * n * density
    # from this point, all is explained in the related document
    a = -1
    b = 2 * n - 1
    c = n - da
    det = b * b - 4 * a * c
    x = (-b + math.sqrt(det))/(2 * a)

    sdw = round(x)

    dw = 2 * sdw + 1

    if(dw < 0) : dw = 0
    elif(dw > 2*n - 1): dw = 2*n - 1
    # print(f"For matrix of size: {n} and given sparsity: {sparsity}, ideal semi diagonal width is : {x}, chosen dw is {dw}")
    return dw    

def get_window_attention_mask_with_sparsity( size, sparsity):
    # conditions : 0 <= sparsity <= 1
    dw = best_diagonal_width_from_sparsity(size, sparsity)
    return get_window_attention_mask( size=size, diagonal_width=dw)

def get_global_attention_mask( size, global_width):
    mask = np.zeros(shape=(size,size), dtype=bool)
    mask[:global_width,:] = True
    mask[global_width : , : global_width] = True
    return mask

def generate_matrix_with_global_attention_mask(size, global_width):
    matrix = np.ones((size, size))
    mask = get_global_attention_mask( size=size, global_width=global_width)
    matrix[~mask] = 0
    return matrix

def best_global_width_from_sparsity(size, sparsity):
    n = size
    density = 1.0 - sparsity
     # ideal diagonal aera
    ga = n * n * density
    # same as window mask but easier
    a = -1
    b = 2 * n
    c = - ga
    det = b * b - 4 * a * c
    x = (-b + math.sqrt(det))/(2 * a)
    gw = round(x)
    if(gw < 0) : gw = 0
    elif(gw > n * n ): gw = n * n
    return gw 

def get_global_attention_mask_with_sparsity( size, sparsity):
    # conditions : 0 <= sparsity <= 1
    gw = best_global_width_from_sparsity(size, sparsity)
    return get_global_attention_mask( size=size, global_width=gw)

# ## BIG BIRD (combination of all above)

def get_big_bird_mask(size, nz_per_row, diagonal_width, global_width):
    am = get_random_attention_mask( size= size, nz_per_row=nz_per_row)
    wm = get_window_attention_mask( size= size, diagonal_width=diagonal_width)
    gm = get_global_attention_mask( size=size, global_width= global_width)
    total_mask = am | wm | gm 
    return total_mask

def generate_big_bird(size, nz_per_row, diagonal_width, global_width ):
    matrix = np.ones((size, size))
    mask = get_big_bird_mask( size=size, nz_per_row= nz_per_row, diagonal_width=diagonal_width, global_width= global_width)
    matrix[~mask] = 0
    return matrix

def get_big_bird_mask_with_sparsity( size, random_sparsity, window_sparsity, global_sparsity):
    am = get_random_attention_mask_with_sparsity( size= size, sparsity=random_sparsity)
    wm = get_window_attention_mask_with_sparsity( size= size,sparsity=window_sparsity )
    gm = get_global_attention_mask_with_sparsity(size=size, sparsity=global_sparsity)
    total_mask = am | wm | gm 
    return total_mask

def adjust_total_sparsity(total_sparsity):
    x = total_sparsity
    # degree = 3
    # a = 2.61815675
    # b = -4.77052715
    # c = 2.98999146
    # d = 0.19945692
    # res =  a  * ( x ** 3 )  + b * (x ** 2) + c * x
    # degree = 5
    a = 24.08862473
    b = -65.2963488
    c = 64.48601296
    d = -28.42365239 
    e = 5.98076684
    f = 0.17082526
    poly =  a * x**5 + b * x**4 + c * x**3 + d * x**2 + e * x + f
    res = min(max(poly, 0.0), 1.0)
    return res

def get_big_bird_mask_with_total_sparsity( size, total_sparsity, adjust):
    if adjust :
        total_sparsity = adjust_total_sparsity(total_sparsity)
    random_sparsity = total_sparsity
    window_sparsity = total_sparsity
    global_sparsity = total_sparsity
    total_mask = get_big_bird_mask_with_sparsity( size, random_sparsity, window_sparsity, global_sparsity)
    return total_mask

def generate_big_bird_with_total_sparsity(size,total_sparsity, adjust):
    matrix = np.ones((size, size))
    mask = get_big_bird_mask_with_total_sparsity(size, total_sparsity, adjust)
    matrix[~mask] = 0
    return matrix

# END OF NOTEBOOK

# Generated by AI

from matplotlib.widgets import Slider

def generate_test_matrices():
    """
    Generates a list of matrices with varying sizes and sparsities.
    Uses the adjusted sparsity function for accurate results.
    """
    # Define the parameters for generation
    sizes = [10, 25, 50, 100, 250, 500, 1000]
    sparsity_values = [x / 100.0 for x in range(0, 101, 5)]

    generated_data = []

    # Loop through each combination and generate the matrix
    for size in sizes:
        for sparsity in sparsity_values:
            # Generate the matrix with adjust=True for better accuracy
            matrix = generate_big_bird_with_total_sparsity(size, sparsity, adjust=True)
            # Pre-calculate the real sparsity
            real_sparsity = get_sparsity(matrix, size)
            # Store matrix, size, given sparsity, and real sparsity
            generated_data.append((matrix, size, sparsity, real_sparsity))

    return generated_data


def interactive_final_test(matrix_data):
    """
    Creates an interactive matplotlib plot to visualize the preloaded matrices.
    Uses sliders to select size and sparsity.
    """
    # Restructure data for quick O(1) access: {size: {sparsity: (matrix, real_sparsity)}}
    data_map = {}
    for matrix, size, given_sparsity, real_sparsity in matrix_data:
        if size not in data_map:
            data_map[size] = {}
        # Store both the matrix and its pre-calculated real sparsity
        data_map[size][f'{given_sparsity:.2f}'] = (matrix, real_sparsity)

    sizes = sorted(data_map.keys())
    sparsities = sorted([float(s) for s in data_map[sizes[0]].keys()])

    # --- Initial Setup ---
    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(bottom=0.25) # Make space for sliders

    initial_size = sizes[0]
    initial_sparsity = sparsities[0]
    initial_matrix, _ = data_map[initial_size][f'{initial_sparsity:.2f}']

    # Display the initial matrix with black & white colormap and fixed 0-1 range
    img = ax.imshow(initial_matrix, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
    fig.colorbar(img, ax=ax)

    # --- Sliders ---
    ax_size = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_sparsity = plt.axes([0.25, 0.05, 0.65, 0.03])

    # Slider for size (uses indices to handle non-uniform steps)
    s_size = Slider(ax_size, 'size', 0, len(sizes) - 1, valinit=0, valstep=1)
    s_size.valtext.set_text(sizes[0]) # Display actual size value

    # Slider for sparsity
    s_sparsity = Slider(ax_sparsity, 'Sparsity', sparsities[0], sparsities[-1], valinit=initial_sparsity, valstep=0.05)

    # --- Update Function ---
    def update(val):
        # Get current values from sliders
        size_idx = int(s_size.val)
        current_size = sizes[size_idx]
        s_size.valtext.set_text(current_size) # Update slider text

        # Find the closest sparsity value available
        current_sparsity_val = min(sparsities, key=lambda x: abs(x - s_sparsity.val))

        # Retrieve the matrix and its pre-calculated real sparsity by unpacking the tuple
        matrix, real_sparsity = data_map[current_size][f'{current_sparsity_val:.2f}']

        # Update plot data and title
        img.set_data(matrix)
        ax.set_title(f"size: {current_size}, Given Sparsity: {current_sparsity_val:.2f}, Real Sparsity: {real_sparsity:.4f}")

        fig.canvas.draw_idle()

    # Register the update function to be called on slider change
    s_size.on_changed(update)
    s_sparsity.on_changed(update)

    # Trigger initial title update
    update(None)

    plt.show()


# Run the interactive test with the preloaded data

def main():
    preloaded_data = generate_test_matrices()
    print(f"Generated and preloaded {len(preloaded_data)} matrices into memory.")
    interactive_final_test(preloaded_data)
    
    
if __name__=="__main__":
    main()

