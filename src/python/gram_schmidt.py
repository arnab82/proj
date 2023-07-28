import numpy as np

"""The Gram-Schmidt process, also known as the Gram-Schmidt method or orthogonalization, is a mathematical procedure used to transform a set of linearly independent vectors 
into an orthogonal (orthonormal) set. This process is widely used in linear algebra, signal processing, and various numerical algorithms.

Given a set of linearly independent vectors {v₁, v₂, ..., vₖ} in an inner product space (such as a vector space with a dot product defined), the Gram-Schmidt process 
constructs an orthogonal set {u₁, u₂, ..., uₖ} such that each uᵢ is orthogonal to all previous vectors, and their span is the same as the original set of vectors.

The Gram-Schmidt process is defined as follows for each vector vᵢ:

1. Initialize the first vector u₁ as the normalized version of v₁: u₁ = v₁ / ||v₁||, where ||v₁|| is the norm (length) of v₁.

2. For each subsequent vector vᵢ (i > 1):
   a. Project vᵢ onto each previously obtained orthogonal vector uⱼ (for j = 1 to i-1) using the dot product:
      proj = (vᵢ ⋅ uⱼ) / (uⱼ ⋅ uⱼ) * uⱼ
   b. Subtract the projections from vᵢ to make it orthogonal to all previous vectors:
      uᵢ = vᵢ - ∑(proj)

3. Normalize the resulting vector uᵢ: uᵢ = uᵢ / ||uᵢ||

After completing the Gram-Schmidt process for all vectors vᵢ, the resulting set of vectors {u₁, u₂, ..., uₖ} is an orthogonal basis for the subspace spanned
   by the original vectors {v₁, v₂, ..., vₖ}. If we further normalize these vectors, we obtain an orthonormal basis for the subspace.

The Gram-Schmidt process is essential for many applications, such as solving linear systems, least squares problems, finding eigenvalues and eigenvectors, and
   various other numerical algorithms that require orthogonalization or orthonormalization of vectors."""
def unit_vec(x):
    #   arg: A vector.
    #   return: The unitvector of a given vector.
    a=x/np.linalg.norm(x)
    return a
def proj(x,subspace):
    #   arg: A vector and a subspace.
    #   return: The projection of the vector on the subspace.
    dim_subspace=len(subspace)
    projx=0
    for l in range(dim_subspace):
        projx+=(np.dot(x,subspace[l]))*subspace[l]
    return projx
def gram_schmidt(vec_list):
    #   arg : A list of linearly independent vectors.
    #   return: A list of orthonormalized vectors of the same span.
    k=len(vec_list)                                                             #There are k linearly independent vectors.
    u=[]                                                                        #List of orthonormal basis vectors (output) appended.
    u.append(unit_vec(vec_list[0]))
    y=[]
    y.append(unit_vec(vec_list[0]))
    for n in range(1,k):
        y.append(vec_list[n]-proj(vec_list[n],u))
        u.append(unit_vec(y[n]))
    return u