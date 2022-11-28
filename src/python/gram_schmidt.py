import numpy as np
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