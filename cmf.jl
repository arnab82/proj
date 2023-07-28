using LinearAlgebra
using SparseArrays
using PyCall
using Einsum
using Printf
function get_cluster_eri(bl, h, g)
    """This code defines a function get_cluster_eri that takes bl (a list or array),
     h (a 2D matrix), and g (a 4D tensor) as input. It initializes ha and ga as arrays of zeros 
     with appropriate dimensions. Then, it iterates over the elements of bl and fills 
     in the corresponding values in ha and ga using nested loops."""
    size_bl = length(bl)
    ha = zeros(size_bl, size_bl)
    ga = zeros(size_bl, size_bl, size_bl, size_bl)

    #AAAA
    for (i, a) in enumerate(bl)
        for (j, b) in enumerate(bl)
            ha[i, j] = h[a, b]
            for (k, c) in enumerate(bl)
                for (l, d) in enumerate(bl)
                    ga[i, j, k, l] = g[a, b, c, d]
                end
            end
        end
    end

    return ha, ga
end
function get_block_eri_2(block, Cluster, tei, a, b, c, d)
    """
    Gives the two electron integral living in respective blocks <AB|CD>
    """
    """This code defines a function get_block_eri_2 that takes block (not used in the function),
     Cluster (an array of Cluster objects), tei (a 4D tensor), and a, b, c, d as input.
      It initializes g_bl as an array of zeros with the appropriate dimensions. 
      Then, it iterates over the elements of Cluster[a].orb_list, Cluster[b].orb_list, Cluster[c].orb_list, 
    and Cluster[d].orb_list and fills in the corresponding values in g_bl using nested loops."""
    g_bl = zeros(Cluster[a].n_orb, Cluster[b].n_orb, Cluster[c].n_orb, Cluster[d].n_orb)

    for (i, I) in enumerate(Cluster[a].orb_list)
        for (j, J) in enumerate(Cluster[b].orb_list)
            for (k, K) in enumerate(Cluster[c].orb_list)
                for (l, L) in enumerate(Cluster[d].orb_list)
                    g_bl[i, j, k, l] = tei[I, J, K, L]
                end
            end
        end
    end

    return g_bl
end
function form_Heff(blocks, Cluster, tei)
    """This code defines a function form_Heff that takes blocks (an array), Cluster (an array of Cluster objects),
     and tei (a 4D tensor) as input. It initializes VJa and VKa as dictionaries to store the results. 
     Then, it iterates over the indices a and b and calculates the values for VJa[a, b] and VKa[a, b] 
     by calling the get_block_eri_2 function and performing the necessary matrix multiplications using the sum function.
"""
    n_blocks = length(blocks)
    VJa = Dict()
    VKa = Dict()
    for a in 1:n_blocks
        for b in 1:n_blocks
            if b != a
                gaabb = get_block_eri_2(blocks, Cluster, tei, a, a, b, b)
                gabba = get_block_eri_2(blocks, Cluster, tei, a, b, b, a)
                Jtemp = einsum("prqs,qs->pr", gaabb, Cluster[b].tdm["ca_aa"])
                Ktemp = einsum("psqr,qs->pr", gabba, Cluster[b].tdm["ca_aa"])
                # Jtemp = sum(gaabb .* Cluster[b].tdm["ca_aa"], dims=(3, 4))
                # Ktemp = sum(gabba .* Cluster[b].tdm["ca_aa"], dims=(3, 4))

                VJa[(a, b)] = Jtemp#coulomb integral
                VKa[(a, b)] = Ktemp#exchange integral
            end
        end
    end

    return VJa, VKa
end
function run_cmf_iter(blocks, Cluster, tei)
    """This code defines a function run_cmf_iter that takes blocks (an array), Cluster (an array of Cluster objects),
     and tei (a 4D tensor) as input. It initializes hnew and mat as dictionaries to store intermediate results. 
     bl_vec is also initialized as a dictionary. 
     The variable EE is initialized as 0.0.
    The code then iterates over the indices a from 1 to n_blocks. Inside the loop, it calculates 
    the effective Hamiltonian heff by summing the contributions from VJa and VKa matrices. 
    The FCI calculation is performed using the fci.direct_spin0.FCI() module, and the resulting CI vectors and energies are obtained"""
    n_blocks = length(blocks)

    VJa, VKa = form_Heff(blocks, Cluster, tei)

    hnew = Dict()
    mat = Dict()

    bl_vec = Dict()

    EE = 0.0
    for a in 1:n_blocks
        damp = false

        if damp
            dd = Cluster[a].p_evec' * Cluster[a].p_evec
            println(dd)
        end

        heff = copy(Cluster[a].oei)
        for b in 1:n_blocks
            if a != b
                heff += 2 * VJa[(a, b)] - VKa[(a, b)]#effective hamiltonian
            end
        end
        # println("effective hamiltonian")
        # display(heff)
        norb_a = Cluster[a].n_orb
        ha = Cluster[a].oei#one-electron part
        ga = Cluster[a].tei#two-electron part

        n_a = Cluster[a].n_a#alpha electrons
        n_b = Cluster[a].n_b#beta electrons

        if n_a + n_b != 0
            fci=pyimport("pyscf.fci")

            cisolver = fci.direct_spin0.FCI()
            efci, ci = cisolver.kernel(heff, ga, norb_a, (n_a, n_b), ecore=0, nroots=1, verbose=100)
            println(efci)

            fci_dim = size(ci, 1) * size(ci, 2)
            ci = reshape(ci, (1, fci_dim))
        else
            efci, ci = cisolver.kernel(heff, ga, norb_a, (n_a, n_b), ecore=0, nroots=1, verbose=6)
            println(efci)

            fci_dim = size(ci, 1) * size(ci, 2)
            ci = reshape(ci, (1, fci_dim))
        end
        #make the density matrix
        d1, d2 = cisolver.make_rdm1s(ci, size(ha, 2), (n_a, n_b))
        Cluster[a].store_tdm("ca_aa", d1)
        Cluster[a].store_tdm("ca_bb", d2)

        if damp
            cv = reshape(ci, (1, fci_dim))
            d2 = cv' * cv
            ee, ci = eigen(0.5 * d2 + 0.5 * dd)
        end

        Cluster[a].tucker_vecs(ci, efci)#only one P vector is considered

        EE += efci

        println("Cluster  : ", a)
        println("     Energy                      : ", efci)
    end

    return EE, bl_vec
end
function double_counting(blocks, Cluster, eri)
    """This code defines a function double_counting that takes blocks (an array), Cluster (an array of Cluster objects), 
    and eri (a 4D tensor) as input. It initializes e_double, e_d, e_extra2, E1_cmf, and E2_cmf as 0.0. 
    The variable Eij is also initialized as 0.0.
    The code then iterates over the indices a from 1 to n_blocks and b from 1 to a-1. Inside the loop, it calculates 
    the double-counting contribution Eij by summing the products of the relevant two-electron integrals 
    gaabb and the corresponding density matrices Pi and Pj. The results are stored in the e_d matrix.
    The final value of Eij is returned."""
    n_blocks = length(blocks)
    e_double = 0.0
    e_d = zeros(n_blocks, n_blocks)
    e_extra2 = 0.0
    E1_cmf = 0.0
    E2_cmf = 0.0
    Eij = 0.0

    for a in 1:n_blocks
        for b in 1:a-1
            if a != b
                gaabb = get_block_eri_2(blocks, Cluster, eri, a, a, b, b)
                gabba = get_block_eri_2(blocks, Cluster, eri, a, b, b, a)

                Pi = Cluster[a].tdm["ca_aa"]
                Pj = Cluster[b].tdm["ca_aa"]
                @einsum Eij+=gaabb[p,q,r,s]*Pi[p,q]*Pj[r,s]
                @einsum Eij-=gabba[p,s,r,q]*Pi[p,q]*Pj[r,s]
                # Eij += sum(gaabb .* Pi .* Pj)
                # Eij -= sum(gabba .* Pi .* Pj)

                Pi = Cluster[a].tdm["ca_bb"]
                Pj = Cluster[b].tdm["ca_bb"]
                @einsum Eij+=gaabb[p,q,r,s]*Pi[p,q]*Pj[r,s]
                @einsum Eij-=gabba[p,s,r,q]*Pi[p,q]*Pj[r,s]
                # Eij += sum(gaabb .* Pi .* Pj)
                # Eij -= sum(gabba .* Pi .* Pj)

                Pi = Cluster[a].tdm["ca_aa"]
                Pj = Cluster[b].tdm["ca_bb"]
                @einsum Eij+=gaabb[p,q,r,s]*Pi[p,q]*Pj[r,s]
                # Eij += sum(gaabb .* Pi .* Pj)

                Pi = Cluster[a].tdm["ca_bb"]
                Pj = Cluster[b].tdm["ca_aa"]
                @einsum Eij+=gaabb[p,q,r,s]*Pi[p,q]*Pj[r,s]
                # Eij += sum(gaabb .* Pi .* Pj)

                e_d[a, b] = Eij
            end
        end
    end

    return Eij
end
function run_cmf(h, g, blocks, fspace, ecore, miter)
    """
    run_cmf(h, g, blocks, fspace, ecore=0, miter=50)

    Runs the Cluster Mean Field (CMF) method for a given Hamiltonian and interaction tensor.

    # Arguments
    - `h::Array{Float64,2}`: The one-body Hamiltonian matrix.
    - `g::Array{Float64,4}`: The two-body interaction tensor.
    - `blocks::Array{Array{Int64,1},1}`: An array of orbital block indices.
    - `fspace::Array{Tuple{Int64,Int64},1}`: An array of tuples representing the number of occupied orbitals in each block.
    - `ecore::Float64=0`: The nuclear repulsion energy (default: 0).
    - `miter::Int64=50`: The maximum number of CMF iterations (default: 50).

    # Returns
    - `Ecmf::Float64`: The CMF energy."""
    """The function initializes variables and prints initial information, such as the number of blocks and the block indices.
    It initializes cluster objects for each block by calculating the local Hamiltonian within each cluster.
    The function computes the one-cluster terms and stores the results in the cluster objects. This includes performing diagonalization of each cluster's local Hamiltonian, computing cluster energies, and storing transformed density matrices.
    It calculates the double-counting contribution by calling the double_counting function, which calculates the double-counting terms in the mean-field energy.
    The initial CMF energy is computed by summing the cluster energies, adding the nuclear repulsion energy, and subtracting the double-counting contribution.
    In each iteration, the CMF energy from the previous iteration is stored as EE_old, and a new iteration is performed by calling the run_cmf_iter function. The resulting CMF energy and block vectors are returned.
    The iteration number, energy from the previous iteration, and energy difference are printed.
    If the energy difference between iterations is below a threshold (1E-8), indicating convergence, the CMF energy is updated by subtracting the double-counting contribution and printing the final results. The variable converged is set to true.
    If the convergence criterion is not met, the double-counting contribution is recalculated, and the CMF energy is updated accordingly.
    If the maximum number of iterations is reached (miter is 0), the function prints a message indicating that the energy for the reference state has been computed but the CMF did not converge.
    Finally, the function returns the CMF energy.
    
"""    
    println("-------------------------------------------------------------------")
    println("                     Start of n-body Tucker ")
    println("-------------------------------------------------------------------")
    n_blocks = length(blocks)
    Ecmf = 0.0
    n_orb = size(h, 1)

    @assert n_blocks == length(blocks)
    size_blocks = [length(i) for i in blocks]

    println("\nNumber of Blocks                               :%10i" , n_blocks)
    println("\nBlocks :", blocks, "\n")

    #initialize the cluster class
    println("Initialize the clusters:")
    cluster = Dict{}()
    for a in 1:n_blocks
        n_elec=init_fspace[a]
        ha, ga = get_cluster_eri(blocks[a], h, g)  #Form integrals within a cluster
        cluster[a]=Cluster(blocks[a],ha,ga,size(ha, 1),zeros(size(ha, 1),size(ha, 1)),zeros(size(ha, 1)),0,init_fspace[a][1],init_fspace[a][2],0.0,0.0,Dict{},0.0)
        println(cluster[a])
        println(ha)
    end

    ###One cluster Terms
    EE = 0.0
    d1 = Dict{Any, Any}()

    for a in 1:n_blocks
        norb_a = cluster[a].n_orb
        ha = cluster[a].oei
        ga = cluster[a].tei

        n_a = fspace[a][1]
        n_b = fspace[a][2]

        if 0
            H = run_fci(ha, ga, norb_a, n_a, n_b)
            S2 = form_S2(norb_a, n_a, n_b)
            efci2, ci2 = sparse.linalg.eigsh(H + S2, k = 4, which = "SA")
            println("ham")
            println(H)
            println(efci2)
            #println(ci.T @ H @ ci)
            #println(ci.T @ S2 @ ci)
            #println(ci)
            #ci = ci.reshape(nCr(norb_a,n_a),nCr(norb_a,n_b))
        end

        if 1
            if n_a + n_b != 0
                fci=pyimport("pyscf.fci")
                cisolver = fci.direct_spin0.FCI()
                efci, ci = cisolver.kernel(ha, ga, norb_a, (n_a, n_b), ecore = 0, nroots = 1, verbose = 100)
                println(efci)
                #efci = efci[1]
                #ci = ci[1]
                fci_dim = size(ci, 1) * size(ci, 2)
                ci = reshape(ci, (1, fci_dim))
            else
                efci, ci = cisolver.kernel(ha, ga, norb_a, (n_a, n_b), ecore = 0, nroots = 1, verbose = 6)
                #println(ci)
                println(efci)
                fci_dim = size(ci, 1) * size(ci, 2)
                ci = reshape(ci, (1, fci_dim))
            end

            ## fci
            cluster[a].tucker_vecs(ci, efci) #Only one P vector right now

            ## make tdm
            d1, d2 = cisolver.make_rdm1s(ci, size(ha, 2), (n_a, n_b))
            cluster[a].store_tdm("ca_aa", d1)
            cluster[a].store_tdm("ca_bb", d2)
        end

        println("Diagonalization of each cluster local Hamiltonian    %16.8f:" % efci)

        EE += efci
        @printf( "cluster: %6d" , a)
        @printf(" Energy: %16.10f" , efci)
    end

    e_double = double_counting(blocks, cluster, g)
    @printf("Ground State nbT-0                             :%16.10f" ,(EE + ecore + e_double))

    EE_old = EE
    Ecmf = EE + ecore + e_double
    converged = false

    println("\nBegin cluster Optimisation...\n")
    #println("    Iter                     Energy                    Error ")
    #println("-------------------------------------------------------------------")
    for i in 0:miter
        EE_old = EE
        println("-------------------------------------------------------------------")
        @printf(" CMF Iteration            : %4d" , i)
        println("-------------------------------------------------------------------")

        EE, bl_vec = run_cmf_iter(blocks, cluster, g)

        @printf("  %4i    Energy: %16.12f         Error:%16.12f" ,i, EE_old, EE - EE_old)
        
        if abs(EE - EE_old) < 1E-8
            println("")
            println("CMF energy converged...")

            e_double = double_counting(blocks, cluster, g)

            @printf("Sum of Eigenvalues                   :%16.10f" , EE)
            @printf("Nuclear Repulsion                    :%16.10f " , ecore)
            @printf("Removing Double counting...          :%16.10f" ,e_double)
            println("")
            Ecmf = EE + ecore - e_double
            #@printf("SCF                                         :%16.10f",Escf)
            @printf("CMF                                         :%16.10f" ,Ecmf)
            println("")
            converged = true
            break
        else
            e_double = double_counting(blocks, cluster, g)
            @printf("Sum of Eigenvalues                   :%16.10f" ,EE)
            @printf("Nuclear Repulsion                    :%16.10f ", ecore)
            @printf("Removing Double counting...          :%16.10f" ,e_double)
            println("")
            Ecmf = EE + ecore - e_double
            #@printf("SCF                                         :%16.10f",Escf)
            @printf("CMF                                         :%16.10f" ,Ecmf)
        end
    end

    if miter == 0
        println("Energy for the reference state computed")
        @printf("Sum of Eigenvalues                   :%16.10f" ,EE)
        @printf("Nuclear Repulsion                    :%16.10f",ecore)
        @printf("Removing Double counting...          :%16.10f" ,e_double)
        @printf("DPS                                  :%16.10f" ,Ecmf)
        println(" -------CMF did not converge--------")
    elseif converged == false
        println(" -------CMF did not converge--------")
    end



    #wfn.Ca().copy(psi4.core.Matrix.from_array(C))
    #wfn.Cb().copy(psi4.core.Matrix.from_array(C))

    #psi4.molden(wfn,"psi4.molden")

    return Ecmf
end


function lowdin(S)
    """the eig function is used to compute the eigenvalues and eigenvectors of a matrix.
     The sortperm function is used to obtain the indices that would sort the eigenvalues in descending order. 
     The diagm function is used to create a diagonal matrix from a vector of diagonal elements."""
    println("Using lowdin orthogonalized orbitals")
    # Forming S^-1/2 to transform to A and B block.
    sal, svec = eig(S)
    idx = sortperm(sal, rev=true)
    sal = sal[idx]
    svec = svec[:, idx]
    sal = diagm(sal .^ -0.5)
    X = svec * sal * svec'
    return X
end
