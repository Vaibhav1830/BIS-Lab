import numpy as np
from scipy.signal import convolve2d

def update_forest(forest):
    """Perform one parallel update step for the forest fire model."""
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]])
    burning = (forest == 1)
    burning_neighbors = convolve2d(burning, kernel, mode='same', boundary='wrap')
    new_forest = np.copy(forest)
    new_forest[forest == 1] = 2  # burning -> burned
    new_forest[(forest == 0) & (burning_neighbors > 0)] = 1  # tree ignites
    return new_forest

def initialize_forest(size=10, tree_density=0.9, fire_density=0.05):
    """
    Smaller grid for easier viewing.
    States:
      0 = Tree (green)
      1 = Burning (red)
      2 = Burned (black)
    """
    forest = np.zeros((size, size), dtype=np.uint8)
    random_vals = np.random.rand(size, size)
    forest[random_vals > tree_density] = 2
    forest[random_vals < fire_density] = 1
    return forest

# ============================
# MAIN PROGRAM
# ============================
np.random.seed(42)
size = 10   # smaller for readability
steps = 10  # number of time steps

forest = initialize_forest(size=size, tree_density=0.9, fire_density=0.05)

print("Initial Forest (Step 0):")
print(forest)

for t in range(steps):
    forest = update_forest(forest)

print("\nFinal Forest (Step", steps, "):")
print(forest)
