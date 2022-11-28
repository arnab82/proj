r=18
n=4
using Einsum
using PyCall
np=PyCall.pyimport("numpy")
using LinearAlgebra

function normalise(mat)
    nrow,ncol=size(mat)
    normalised_mat=zeros(Float64,nrow,ncol)
    col_element=0.0
    for a in 1:ncol
        for b in 1:nrow
            col_element+=abs((mat[b,a])^2)
        end
        for b in 1:nrow
            normalised_mat[b,a]=mat[b,a]/(sqrt(col_element))       
        end
        col_element=0.0   
    end
    return normalised_mat
end

function tensor_decom()
    A=rand(Float64,n,n,n,n)
    a=rand(Float64,n,r)
    b=rand(Float64,n,r)
    c=rand(Float64,n,r)
    d=rand(Float64,n,r)
    a=normalise(a)
    b=normalise(b)
    c=normalise(c)
    d=normalise(d)
    println(size(d))
    error=0.0   
    W=zeros(Float64,n,r)
    V=zeros(Float64,n,r)
    A_=zeros(Float64,n,n,n,n)
    for i in 1:1000
        @einsum W[f,r]=(transpose(b[q,f])*b[q,r])*(transpose(c[s,f])*c[s,r])*(transpose(d[t,f])*d[t,r])   
        @einsum V[r,p]=A[p,q,s,t]*transpose(b[q,r])*transpose(c[s,r])*transpose(d[t,r])
        #display(V)
        #display(W)
        a=transpose(W\V)
        a=a[1:n,:]
        #a=transpose(\(W,V))#why it is  12 x 12 
        display(a[1:n,:])
        a=normalise(a)
        @einsum W[f,r]=(transpose(a[p,f])*a[p,r])*(transpose(c[s,f])*c[s,r])*(transpose(d[t,f])*d[t,r])   
        @einsum V[r,q]=A[p,q,s,t]*transpose(a[p,r])*transpose(c[s,r])*transpose(d[t,r])
        b=transpose(W\V)
        b=b[1:n,:]
        b=normalise(b)
        @einsum W[f,r]=(transpose(a[p,f])*a[p,r])*(transpose(b[q,f])*b[q,r])*(transpose(d[t,f])*d[t,r])   
        @einsum V[r,s]=A[p,q,s,t]*transpose(a[p,r])*transpose(b[q,r])*transpose(d[t,r])
        c=transpose(W\V)
        c=c[1:n,:]
        c=normalise(c)
        @einsum W[f,r]=(transpose(a[p,f])*b[p,r])*(transpose(b[q,f])*c[q,r])*(transpose(c[s,f])*c[s,r])   
        @einsum V[r,t]=A[p,q,s,t]*transpose(a[p,r])*transpose(b[q,r])*transpose(c[s,r])
        d=transpose(W\V)
        d=d[1:n,:]
        @einsum A_[p,q,s,t]=a[p,r]*b[q,r]*c[s,r]*d[t,r]
        d=normalise(d)
        new_error=0.5.*(sum(abs.((A_-A).^2)))
        display(i)
        println(new_error)
        del_err=new_error-error
         
        if abs(new_error)<=(1e-15)
            break
        else
            new_error=error
        end
        A_=A
    end
    print("the decomposition is successful","\n")
    print("the value of d is",d)
    print("the value of a is",a)
    print("the value of b is",b)
    print("the value of c is",c) 
    return a,b,c,d
end         
a,b,c,d=tensor_decom()       
