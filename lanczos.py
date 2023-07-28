"""The Lanczos algorithm is an iterative method for finding a few eigenvalues and eigenvectors of a large, symmetric matrix.
It was developed by Cornelius Lanczos in the 1950s and is particularly useful for solving large-scale eigenvalue problems efficiently.
The algorithm belongs to the class of Krylov subspace methods.

Here's a high-level explanation of the Lanczos algorithm:

1. Initialization: Start with a real symmetric matrix A and an initial vector v₀ of length n (where n is the size of the matrix A). Normalize v₀ to have unit length.

2. Iteration: Perform a series of iterations, generating an orthogonal basis for the Krylov subspace spanned by {v₀, Av₀, A²v₀, ..., A^(m-1)v₀}, 
                     where m is the desired number of eigenvalues to compute.

3. Orthogonalization: At each iteration, orthogonalize the current vector against the previous vectors in the Krylov subspace to maintain orthogonality.
                     This can be achieved using the Gram-Schmidt process or a more efficient variant like modified Gram-Schmidt.

4. Tridiagonalization: Construct a tridiagonal matrix T_m of size m × m that captures the essential information about the eigenvalue problem.
                     The entries of T_m are obtained through the Lanczos recurrence relation.

5. Diagonalization: Diagonalize the tridiagonal matrix T_m to obtain the eigenvalues and eigenvectors. This can be done using standard eigenvalue solvers
                     such as the QR algorithm or the symmetric QR algorithm.

6. Convergence: The Lanczos algorithm typically converges rapidly, and after m iterations, it provides an approximation of the m largest or smallest eigenvalues 
                     and their associated eigenvectors.

The Lanczos algorithm is especially useful when computing a small number of eigenvalues, as it constructs a much smaller tridiagonal matrix compared to the original matrix.
                     This reduction in dimensionality significantly reduces computational complexity.The Lanczos algorithm has numerous applications, 
                     including in quantum mechanics, molecular dynamics, and machine learning algorithms such as principal component analysis (PCA) and graph clustering.

"""
import numpy as np

def lanczos(matrix, b, m=12):
    n = matrix.shape[0]
    
    # Initialize empty matrices for T and V
    T = np.zeros((m+1, m), dtype=np.float64)
    V = np.zeros((n, m+1), dtype=np.float64)
    
    # Normalize the starting vector
    V[:, 0] = b / np.linalg.norm(b)
    
    # Next vector
    w = np.matmul(matrix,(V[:, 0]))
    
    # Orthogonalize against the first vector
    T[0, 0] = np.dot(w, V[:, 0])
    w = w - T[0, 0] * V[:, 0]
    
    # Normalize the next vector
    T[1, 0] = np.linalg.norm(w)
    V[:, 1] = w / T[1, 0]
    
    for j in range(1, m):
        # Make T symmetric
        T[j-1, j] = T[j, j-1]
        
        # Next vector
        w = np.matmul(matrix,(V[:, j]))
        
        # Orthogonalize against two previous vectors
        T[j, j] = np.dot(w, V[:, j])
        w = w - np.dot(w, V[:, j]) * V[:, j] - T[j-1, j] * V[:, j-1]
        
        # Normalize
        T[j+1, j] = np.linalg.norm(w)
        V[:, j+1] = w / T[j+1, j]

        # Convergence check
        if T[j+1, j] < 10E-8:
            print(f"\n\nConverged at {j} iteration")
            print("\n", T[j+1, j])
            Tm = T[:j+1, :j+1]
        break
    
    # Make T into a symmetric matrix of shape (m,m)
    Tm = T[:m, :m]
    return Tm, V[:, :m]


n = 20

matrix = np.random.rand(n, n)
b = np.random.rand(n)
Tm, Vm = lanczos(matrix, b, m=12)
eigenvalues = np.diag(Tm)
print(eigenvalues)
print(Vm)