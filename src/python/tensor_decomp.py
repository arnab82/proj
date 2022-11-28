import numpy as np
r=95
n=6
#A= np.arange(r).reshape(n,n)

def normalise(mat):
    nrow,ncol=np.shape(mat)
    normalised_mat=np.zeros(np.shape(mat))
    col_element=0.0
    for a in range(nrow):
        for b in range(ncol):
            col_element+=np.abs((mat[a,b])**2)
        for b in range(ncol):
            normalised_mat[a,b]=mat[a,b]/np.sqrt(col_element)
        col_element=0.0
    return normalised_mat

def tensor_decom():
    A=np.random.random((n,n,n,n))
    a=np.random.random((n,r))
    b=np.random.random((n,r))
    c=np.random.random((n,r))
    d=np.random.random((n,r))
    a=normalise(a)
    b=normalise(b)
    c=normalise(c)
    d=normalise(d)
    print(np.shape(d))
    error=0.0   
    for i in range(1000):
        W=np.einsum("fq,qr,fs,sr,ft,tr->fr",b.T,b,c.T,c,d.T,d)
        V=np.einsum("pqst,rq,rs,rt->rp",A,b.T,c.T,d.T)
        a=np.linalg.solve(W,V).T
        a=normalise(a)
        W=np.einsum("fp,pr,fs,sr,ft,tr->fr",a.T,a,c.T,c,d.T,d)
        V=np.einsum("pqst,rp,rs,rt->rq",A,a.T,c.T,d.T)
        b=np.linalg.solve(W,V).T
        b=normalise(b)
        W=np.einsum("fp,pr,fq,qr,ft,tr->fr",a.T,a,b.T,b,d.T,d)
        V=np.einsum("pqst,rp,rq,rt->rs",A,a.T,b.T,d.T)
        c=np.linalg.solve(W,V).T
        c=normalise(c)
        W=np.einsum("fq,qr,fp,pr,fs,sr->fr",a.T,a,b.T,b,c.T,c)
        V=np.einsum("pqst,rp,rq,rs->rt",A,a.T,b.T,c.T)
        d=np.linalg.solve(W,V).T
        A_=np.einsum("pr,qr,sr,tr->pqst",a,b,c,d)
        d=normalise(d)
        new_error=0.5*np.sum(np.abs((A_-A)**2))
        print(i)
        print(new_error)
        del_err=new_error-error
        A_=A 
        if np.abs(new_error)<=(1e-15):
            break
        else:
            new_error=error
    print("the decomposition is successful")
    print("the value of d is",d)
    print("the value of a is",a)
    print("the value of b is",b)
    print("the value of c is",c) 
    return a,b,c,d           
a,b,c,d=tensor_decom()
