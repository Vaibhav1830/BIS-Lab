import matplotlib.pyplot as plt
import numpy as np

# Coordinates
locations = np.array([
    [0, 0],  # Depot
    [2, 3],
    [5, 4],
    [1, 6],
    [7, 2],
    [6, 6]
])

num_customers = len(locations) - 1

# Calculate Euclidean distance matrix
def distance_matrix(loc):
    n = len(loc)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(loc[i] - loc[j])
    return dist

dist_matrix = distance_matrix(locations)

# Objective function: total route distance (starts and ends at depot)
def route_distance(route):
    dist = dist_matrix[0, route[0]]  # depot to first customer
    for i in range(len(route) - 1):
        dist += dist_matrix[route[i], route[i+1]]
    dist += dist_matrix[route[-1], 0]  # last customer back to depot
    return dist

# Grey Wolf Optimization adapted for VRP (Permutation problem)
class GWO_VRP:
    def __init__(self, population_size, iterations):
        self.population_size = population_size
        self.iterations = iterations
        
        # Initialize wolves with random permutations of customers
        self.population = [np.random.permutation(num_customers) + 1 for _ in range(population_size)]
        self.alpha_pos = None
        self.alpha_score = float('inf')
        self.beta_pos = None
        self.beta_score = float('inf')
        self.delta_pos = None
        self.delta_score = float('inf')

    def fitness(self, wolf):
        return route_distance(wolf)

    def update_leaders(self, wolf):
        fit = self.fitness(wolf)
        if fit < self.alpha_score:
            self.delta_score = self.beta_score
            self.delta_pos = self.beta_pos
            self.beta_score = self.alpha_score
            self.beta_pos = self.alpha_pos
            self.alpha_score = fit
            self.alpha_pos = wolf.copy()
        elif fit < self.beta_score:
            self.delta_score = self.beta_score
            self.delta_pos = self.beta_pos
            self.beta_score = fit
            self.beta_pos = wolf.copy()
        elif fit < self.delta_score:
            self.delta_score = fit
            self.delta_pos = wolf.copy()

    # Permutation-based position update (swap operator inspired by GWO hunting)
    def update_position(self, wolf, leader):
        # To update permutation, swap elements to move wolf closer to leader
        wolf_new = wolf.copy()
        for i in range(num_customers):
            if wolf_new[i] != leader[i]:
                swap_idx = np.where(wolf_new == leader[i])[0][0]
                wolf_new[i], wolf_new[swap_idx] = wolf_new[swap_idx], wolf_new[i]
                break
        return wolf_new

    def optimize(self):
        for _ in range(self.iterations):
            for i in range(self.population_size):
                self.update_leaders(self.population[i])
            
            a = 2 - _ * (2 / self.iterations)  # linearly decreased from 2 to 0
            
            new_population = []
            for wolf in self.population:
                # Update positions towards alpha, beta, delta wolves
                wolf1 = self.update_position(wolf, self.alpha_pos)
                wolf2 = self.update_position(wolf, self.beta_pos)
                wolf3 = self.update_position(wolf, self.delta_pos)
                
                # Select best among the updated wolves
                candidates = [wolf1, wolf2, wolf3]
                fitnesses = [self.fitness(w) for w in candidates]
                best_idx = np.argmin(fitnesses)
                new_population.append(candidates[best_idx])
            
            self.population = new_population
        
        return self.alpha_pos, self.alpha_score

# Running the optimizer
gwo = GWO_VRP(population_size=10, iterations=100)
best_route, best_distance = gwo.optimize()

print("Best route found:", best_route)
print("Best distance:", best_distance)

route = best_route

plt.figure(figsize=(8,6))
plt.scatter(locations[0,0], locations[0,1], c='blue', label='Depot', s=100)
plt.scatter(locations[1:,0], locations[1:,1], c='red', label='Customers', s=100)

# Plot route lines
route_coords = [locations[0]] + [locations[i] for i in route] + [locations[0]]
route_coords = np.array(route_coords)

plt.plot(route_coords[:,0], route_coords[:,1], linestyle='-', marker='o', color='green')

for i, (x,y) in enumerate(locations):
    if i == 0:
        plt.text(x, y+0.3, 'Depot', fontsize=12, fontweight='bold')
    else:
        plt.text(x, y+0.3, f'C{i}', fontsize=12)

plt.title("Vehicle Routing Problem Example")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.legend()
plt.grid(True)
plt.show()
