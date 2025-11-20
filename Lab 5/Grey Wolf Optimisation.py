import cv2
import numpy as np

# Step 1: Define the problem (fitness function)
def otsu_fitness(image, threshold):
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    total = image.size
    sum_total = np.dot(np.arange(256), hist)
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(np.arange(256) * hist)
    mean_bg = sum_bg / np.maximum(weight_bg, 1)
    mean_fg = (sum_total - sum_bg) / np.maximum(weight_fg, 1)
    var_between = weight_bg[:-1] * weight_fg[:-1] * (mean_bg[:-1] - mean_fg[:-1])**2
    if int(threshold) >= len(var_between):
        return 0
    return var_between[int(threshold)]

# Step 2: Initialize parameters
num_wolves = 20
max_iter = 50
lower_bound = 0
upper_bound = 255

# Step 3: Load image and initialize population
image = cv2.imread(r"C:\Users\BMSCE\Downloads\test_image.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Check path or filename.")
wolves = np.random.uniform(lower_bound, upper_bound, num_wolves)

# Step 4: Evaluate fitness
fitness = np.array([otsu_fitness(image, w) for w in wolves])
alpha, beta, delta = wolves[np.argsort(-fitness)[:3]]
alpha_score, beta_score, delta_score = np.sort(fitness)[-3:][::-1]

# Step 6: Iterate
for t in range(max_iter):
    a = 2 - 2 * (t / max_iter)
    for i in range(num_wolves):
        r1, r2 = np.random.rand(), np.random.rand()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha - wolves[i])
        X1 = alpha - A1 * D_alpha

        r1, r2 = np.random.rand(), np.random.rand()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta - wolves[i])
        X2 = beta - A2 * D_beta

        r1, r2 = np.random.rand(), np.random.rand()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta - wolves[i])
        X3 = delta - A3 * D_delta

        wolves[i] = np.clip((X1 + X2 + X3) / 3, lower_bound, upper_bound)

    fitness = np.array([otsu_fitness(image, w) for w in wolves])
    best_indices = np.argsort(-fitness)[:3]
    alpha, beta, delta = wolves[best_indices]
    alpha_score, beta_score, delta_score = fitness[best_indices]

# Step 7: Output best solution
best_threshold = alpha
segmented = (image >= best_threshold).astype(np.uint8) * 255

print(f"Optimal Threshold Found: {best_threshold}")
cv2.imwrite("segmented_image.jpg", segmented)


