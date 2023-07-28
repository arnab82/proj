import numpy as np
from gram_schmidt import *


n = 1200					# Dimension of matrix
sparsity = 0.000001  
A = np.zeros((n,n))  
for i in range(0,n):  
    A[i,i] = i + 1  
A = A + sparsity*np.random.randn(n,n)  
A = (A.T + A)/2 
k = 8 # number of initial guess vectors  
eig = 4 # number of eignvalues to solve 
tol = 1e-8				# Convergence tolerance

"""
    Davidson Algorithm for finding a few lowest eigenvalues and eigenvectors of a given matrix.

    Parameters:
        A (numpy.ndarray): The input matrix of shape (n, n).
        n (int): Dimension of the matrix A.
        k (int): Number of initial guess vectors.

    Returns:
        tuple: A tuple containing two elements:
            - numpy.ndarray: A matrix of shape (n, k) containing the eigenvectors.
            - numpy.ndarray: An array of length k containing the corresponding eigenvalues.

    Algorithm Explanation:
        The Davidson algorithm is an iterative method for finding a few lowest eigenvalues and eigenvectors
        of a large sparse symmetric matrix. It starts with a set of k initial guess vectors and updates the
        guess vectors iteratively to converge towards the desired eigenpairs.

        - The initial guess vectors are stored in the matrix 'V', and the matrix 't' is initialized as the
          identity matrix with the first k columns selected as the guess vectors.
        - At each iteration, the matrix 'V' is orthogonalized, and a subspace matrix 'T' is formed by
          projecting 'A' onto the subspace spanned by 'V'. The eigenpairs of 'T' are computed to update the
          guess vectors.
        - The updated guess vectors are used to construct new vectors to expand the subspace.
        - The process is repeated until convergence is achieved or the maximum number of iterations is reached.

    Note:
        The input matrix 'A' should be symmetric or nearly symmetric for accurate results. The parameter 'k'
        determines the number of initial guess vectors, and 'eig' should be adjusted to select the number of
        desired eigenvalues to solve.

    Example:
        # Define the matrix A and set parameters
        n = 1200
        k = 8
        sparsity = 0.000001
        A = np.zeros((n, n))
        # ... (populate matrix A, add sparsity, etc.)
        tol = 1e-8

        # Call the davidson function to find eigenvalues and eigenvectors
        eigenvecs, eigenvals = davidson(A, n, k)
        print("Eigenvalues:", eigenvals)
        print("Eigenvectors:", eigenvecs)
    """
def davidson(A,n,k):
    t = np.eye(n,k)			# set of k unit vectors as guess
    V = np.zeros((n,n))		# array of zeros to hold guess vec
    I = np.eye(n)			# identity matrix same dimen as A
    mmax = n//2				# Maximum number of iterations 
    for m in range(k,mmax,k):
        if m <= k:
            for j in range(0,k):
                V[:,j] = t[:,j]/np.linalg.norm(t[:,j])
            theta_old = 1 
        elif m > k:
            theta_old = theta[:eig]
        V[:,:m],R = np.linalg.qr(V[:,:m])
        T = np.dot(V[:,:m].T,np.dot(A,V[:,:m]))
        THETA,S = np.linalg.eig(T)
        idx = THETA.argsort()
        theta = THETA[idx]
        s = S[:,idx]
        for j in range(0,k):
            w = np.dot((A - theta[j]*I),np.dot(V[:,:m],s[:,j])) 
            q = w/(theta[j]-A[j,j])
            V[:,(m+j)] = q
        norm = np.linalg.norm(theta[:eig] - theta_old)
        if norm < tol:
            break
        return np.sort(s),THETA
    
v,eigenvalue=davidson(A,n,k)
print(eigenvalue)
print(v)


