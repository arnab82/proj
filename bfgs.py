import numpy as np
import matplotlib.pyplot as plt
"""The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm is an iterative optimization method used to solve unconstrained nonlinear optimization problems. 
It belongs to the class of quasi-Newton methods, which aim to find the minimum of a function without explicitly computing its derivatives.

Here's an overview of the BFGS algorithm:

1. Initialization:
   - Choose an initial guess for the solution, denoted as x_0.
   - Set an initial approximation of the inverse Hessian matrix, denoted as B_0.
   - Set the iteration counter k = 0.

2. Iteration:
   - Compute the gradient of the objective function at the current point: g_k = ∇f(x_k).
   - If the norm of the gradient is below a specified tolerance, terminate the algorithm (i.e., the current point is close to the minimum).
   - Search direction: p_k = -B_k * g_k, where B_k is the current approximation of the inverse Hessian matrix.
   - Perform a line search to determine the step size or "learning rate" that minimizes the objective function along the search direction. Denote the step size as α_k.
   - Update the solution: x_{k+1} = x_k + α_k * p_k.
   - Compute the new gradient: g_{k+1} = ∇f(x_{k+1}).
   - Compute the change in the gradient: Δg_k = g_{k+1} - g_k.
   - Compute the change in the solution: Δx_k = x_{k+1} - x_k.

3. Update the approximation of the inverse Hessian matrix:
   - Compute the matrix product: y_k = Δg_k - B_k * Δx_k.
   - Update the approximation: B_{k+1} = B_k + (y_k * y_k^T) / (y_k^T * Δx_k) - (B_k * Δx_k * Δx_k^T * B_k) / (Δx_k^T * B_k * Δx_k).

4. Increment the iteration counter: k = k + 1, and go to step 2.

The BFGS algorithm iteratively updates the solution and the approximation of the inverse Hessian matrix using the differences in gradients and solutions between iterations.
By approximating the Hessian matrix, it avoids the expensive computation of the exact Hessian and makes the optimization process more efficient.
The algorithm continues until the norm of the gradient falls below a predefined tolerance, indicating that a satisfactory minimum has been found, or until a maximum number 
of iterations is reached.
The BFGS algorithm is widely used for solving optimization problems due to its good convergence properties and efficiency. It has been proven to converge to a local minimum 
for strictly convex functions, and it often performs well on nonconvex problems as well."""



def bfgs_algorithm(initial_x, compute_gradient, max_iterations=1000, tolerance=1e-6):
    # Initialization
    x = initial_x
    n = len(x)
    B = np.eye(n)  # Initial approximation of the inverse Hessian matrix
    k = 0  # Iteration counter

    while k < max_iterations:
        # Step 1: Compute gradient
        g = compute_gradient(x)

        # Step 2: Check convergence
        if np.linalg.norm(g) < tolerance:
            break

        # Step 3: Search direction
        p = -np.dot(B, g)

        # Step 4: Line search
        alpha = line_search(x, p, compute_gradient)  # Updated line search function

        # Step 5: Update solution
        x_next = x + alpha * p

        # Step 6: Compute new gradient and differences
        g_next = compute_gradient(x_next)
        delta_x = x_next - x
        delta_g = g_next - g

        # Step 7: Update approximation of inverse Hessian matrix
        B = update_hessian_approximation(B, delta_x, delta_g)  # Updated Hessian approximation update function
        pe_x=[]
        pe_x.append(x_next)
        # Step 8: Update variables for next iteration
        x = x_next
        k += 1

    return x,pe_x




def line_search(x, p, compute_gradient):
    # Armijo line search method
    alpha = 1.0
    c = 0.5
    rho = 0.5
    g0 = compute_gradient(x)
    phi0 = np.dot(g0, p)
    while True:
        x_next = x + alpha * p
        phi_next = compute_gradient(x_next)
        if np.linalg.norm(phi_next) <= (1 - c * alpha) * np.linalg.norm(g0):
            break
        alpha *= rho
    return alpha




def update_hessian_approximation(B, delta_x, delta_g):
    rho = 1.0 / np.dot(delta_x, delta_g)
    I = np.eye(len(delta_x))
    B_next = (I - np.outer(delta_x, delta_g) * rho).dot(B).dot(I - np.outer(delta_g, delta_x) * rho) + np.outer(delta_x, delta_x) * rho
    return B_next



# Example usage:
def lennard_jones_potential(positions):
    # Assume positions is a numpy array of shape (N, 3) representing N particles' positions in 3D space
    # Compute the Lennard-Jones potential energy based on the positions of the particles

    N = positions.shape[0]
    potential_energy = 0.0

    epsilon = 1.0  # Lennard-Jones potential parameter
    sigma = 1.0  # Lennard-Jones potential parameter

    for i in range(N):
        for j in range(i+1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            potential_energy += 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

    return potential_energy



def compute_gradient(positions):
    # Assume positions is a numpy array of shape (N, 3) representing N particles' positions in 3D space
    # Compute the gradient of the Lennard-Jones potential energy with respect to the particle positions

    N = positions.shape[0]
    gradient = np.zeros_like(positions)

    epsilon = 1.0  # Lennard-Jones potential parameter
    sigma = 1.0  # Lennard-Jones potential parameter

    for i in range(N):
        for j in range(i+1, N):
            r = np.linalg.norm(positions[i] - positions[j])
            direction = (positions[i] - positions[j]) / r
            factor = 24 * epsilon * (2 * (sigma / r)**12 - (sigma / r)**6) / r**2
            gradient[i] += factor * direction
            gradient[j] -= factor * direction

    return gradient



def compute_separation_distances(positions):
    # Compute the separation distances between particles based on their positions
    # Get the number of particles
    N = positions.shape[0]
    # Compute the pairwise differences between positions
    differences = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # Compute the separation distances using the Euclidean norm
    distances = np.linalg.norm(differences, axis=2)
    # Set the diagonal elements to a large value to avoid self-interaction
    np.fill_diagonal(distances, 0)
    return distances


def calculate_distances_from_origin(positions):
    # Calculate the distance of each particle's position from the origin (0, 0, 0)
    # Calculate the Euclidean distance using the Pythagorean theorem
    distances = np.linalg.norm(positions, axis=1)
    return distances
 
def lennard_jones_potential_r(r):
    # Lennard-Jones potential equation
    epsilon = 1.0  # Lennard-Jones potential parameter
    sigma = 1.0  # Lennard-Jones potential parameter
    return 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)

# Example usage:
positions = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
])
solution ,pe_x= bfgs_algorithm(positions.flatten(), compute_gradient)
solution=solution.reshape((-1, 3))
print("Solution:", solution)
separation_distances = compute_separation_distances(solution)
# Compute the energy for each separation distance
energies = lennard_jones_potential_r(separation_distances)

# Plot the energy vs. separation distances
plt.plot(separation_distances.flatten(), energies.flatten())
plt.xlabel('Separation Distance')
plt.ylabel('Potential Energy')
plt.title('Lennard-Jones Potential Energy vs. Separation Distance')
plt.grid(True)
plt.show()

"""
Limited-memory BFGS (L-BFGS) is an optimization algorithm used to solve optimization problems, particularly in cases where the objective function and 
        its gradient are computationally expensive or memory-intensive. L-BFGS belongs to the class of quasi-Newton methods and is an extension of the BFGS algorithm.

The key idea behind L-BFGS is to approximate the inverse Hessian matrix of the objective function using limited memory, instead of storing the full Hessian matrix
        as in the original BFGS method. By utilizing limited memory, L-BFGS reduces the memory requirements and computational complexity associated with the Hessian matrix.

The algorithm maintains a history of past iterations, which includes information about the changes in the position and gradient vectors.
        This history is used to construct an approximation of the inverse Hessian matrix. The approximation is updated iteratively based on the gradient differences between iterations.

Here is a high-level overview of the L-BFGS algorithm:

1. Initialize the position vector `x`, gradient vector `g`, and a limited memory queue (memory buffer) for storing past iterations.
2. While the termination condition is not met:
   a. Compute the search direction by applying the limited-memory BFGS update formula to the gradient vector and the history stored in the memory buffer.
   b. Perform a line search along the search direction to determine an appropriate step size that satisfies the Wolfe conditions.
   c. Update the position vector `x` by taking a step along the search direction with the determined step size.
   d. Compute the new gradient vector `g` at the updated position.
   e. Update the memory buffer with the information about the position and gradient differences.
   f. Check the termination condition, such as reaching a maximum number of iterations or achieving a desired tolerance.
3. Return the final position vector `x` as the solution.

L-BFGS is widely used in various optimization problems, including unconstrained optimization, nonlinear least squares, and training deep neural networks.
        Its memory-efficient nature makes it particularly suitable for large-scale optimization problems where the full Hessian matrix is computationally
        infeasible to store and compute.

The specific implementation details and mathematical formulas involved in L-BFGS can vary, and there are different variations and improvements of the algorithm.
 The core idea, however, remains the approximation of the inverse Hessian matrix using limited memory."""