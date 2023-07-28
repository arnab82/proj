"""Conjugate Gradient is an iterative optimization algorithm used to solve systems of linear equations or to minimize quadratic functions. 
It is particularly useful when dealing with large, sparse, and symmetric positive definite matrices. The key idea behind Conjugate Gradient 
is to find a sequence of mutually conjugate directions that allow for efficient convergence to the solution.

Here's how Conjugate Gradient works:

1. Initialization: Start with an initial guess for the solution vector, x, and compute the initial residual, r0, 
            as the difference between the given right-hand side vector, b, and the matrix-vector product of the system, Ax, where A is the coefficient matrix.
2. Compute the Search Directions: Initialize the search direction, d0, as the negative of the initial residual, d0 = -r0.
3. Iterative Updates:
   a. Compute the step size, αk, by taking the dot product of the current residual, rk, and the search direction, dk, 
                divided by the dot product of the search direction with the matrix-vector product of the system, Adk.
   b. Update the solution vector, x, by adding the step size times the search direction, xk+1 = xk + αk * dk.
   c. Update the residual, rk+1, by subtracting the step size times the matrix-vector product of the system, rk+1 = rk - αk * Adk.
   d. Compute the new search direction, dk+1, by adding the current residual to the previous search direction, dk+1 = -rk+1 + βk * dk, 
                where βk is determined by the ratio of dot products between the new and old residuals.
   e. Repeat steps a-d until a convergence criterion is met, such as reaching a desired tolerance level or a maximum number of iterations.

The conjugate direction property ensures that the search directions are orthogonal to each other with respect to the matrix A, allowing for efficient convergence.
             The algorithm iteratively improves the solution by successively minimizing the residual along mutually conjugate directions.

Conjugate Gradient can be an efficient method for solving large linear systems or minimizing quadratic functions compared to direct methods, 
             especially when dealing with sparse matrices. However, it is primarily designed for symmetric positive definite matrices.
             If the matrix is not symmetric or positive definite, a variant called "Preconditioned Conjugate Gradient" may be used, which introduces a preconditioner to address these cases.

Overall, Conjugate Gradient is a widely used optimization algorithm for solving linear systems and minimizing quadratic functions, providing a computationally efficient 
             and iterative approach for finding solutions."""


import numpy as np

def conjugate_gradient(A, b, x0, max_iterations, tolerance):
    """
    Conjugate Gradient algorithm for solving a linear system Ax = b.

    Parameters:
        A (ndarray): Coefficient matrix of shape (n, n).
        b (ndarray): Right-hand side vector of shape (n,).
        x0 (ndarray): Initial guess for the solution vector of shape (n,).
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        ndarray: Solution vector of shape (n,).
    """

    x = x0
    r = b - A.dot(x)
    d = r
    rsold = r.dot(r)

    for i in range(max_iterations):
        Ad = A.dot(d)
        alpha = rsold / d.dot(Ad)
        x = x + alpha * d
        r = r - alpha * Ad
        rsnew = r.dot(r)
        if np.sqrt(rsnew) < tolerance:
            break
        beta = rsnew / rsold
        d = r + beta * d
        rsold = rsnew

    return x

# Example usage:
# Define the coefficient matrix A, right-hand side vector b, initial guess x0
A = np.array([[10, -3, 2], [-3, 12, -5], [2, -5, 15]])
b = np.array([1, 2, 3])

x0 = np.zeros_like(b)

# Set parameters
max_iterations = 1000
tolerance = 1e-6

# Perform Conjugate Gradient
solution = conjugate_gradient(A, b, x0, max_iterations, tolerance)

print("Solution:", solution)
