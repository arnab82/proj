import numpy as np
from gram_schmidt import *
def davidson_liu(H,k):
    #   arg: A matrix and the number of desired eigenvalues.
    #   return: The eigenvalues and the eigenvectors.
    n=len(H)
    maxiter=100
    for iterations in range(maxiter):
        b=np.eye(n,k)
        sigma=np.matmul(H,b)
        G=np.matmul(np.transpose(b),sigma)
        wG,vG=np.linalg.eigh(G)
        c=np.zeros([n,k])
        r=np.zeros([n,k])
        delta=np.zeros([n,k])
        c=np.einsum("ij,oi->oj",vG,b)
        for xk in range(k):
            pre_r=np.matmul((H-wG[xk]),b[:,xk])
            op=np.transpose(vG[:,xk])
            r=np.matmul(pre_r,op)
        for xn in range(n):
            for xk in range(k):
                delta[xn][xk]=(1/(wG[xk]-H[xn][xn]))*r[xn][xk]
        odelta=gram_schmidt(np.transpose(delta))
        for i in range(len(odelta)):
            if np.linalg.norm(odelta[i])<0.001:
                odelta.remove(odelta[i])
        k=len(odelta)
        b=np.transpose((np.array(odelta)))
        if np.std(delta)<=pow(10,-14):
            break
        elif iterations==maxiter-1:
            print('Maximum iteration cycle reached. Davidson_Liu did not converge after '+str(iterations+1)+' iterations.')
            exit(0)
    print("SUCCESS! Davidson_Liu iteration converged.")
    return wG,vG