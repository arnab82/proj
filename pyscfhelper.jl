using LinearAlgebra
using PyCall

PySCF = pyimport("pyscf")

mutable struct PyscfHelper
    mf::Any
    mol::Any
    h::Array{Float64, 2}
    g::Array{Float64, 4}
    n_orb::Int64
    ecore::Float64
    C::Array{Float64, 2}
    S::Array{Float64, 2}
    J::Array{Float64, 2}
    K::Array{Float64, 2}
    Escf::Float64

    function PyscfHelper()
        new()
    end
end

function lowdin(S)
    println("Using lowdin orthogonalized orbitals")
    sal, svec = eigen(S)
    idx = sortperm(sal, rev=true)
    sal = sal[idx]
    svec = svec[:, idx]
    sal = diagm(sal .^ -0.5)
    X = svec * sal * svec'
    return X
end

function init(helper::PyscfHelper, molecule, charge, spin, basis_set; orb_basis="scf", cas=false, cas_nstart=nothing, cas_nstop=nothing, cas_nel=nothing, loc_nstart=nothing, loc_nstop=nothing)
      # Importing PySCF package
    PySCF.lib.num_threads(1)
    println(" ---------------------------------------------------------")
    println("                      Using Pyscf:")
    println(" ---------------------------------------------------------")
    println("                                                          ")

    mol = PySCF.gto.Mole()
    ao2mo=pyimport("pyscf.ao2mo")
    symm=pyimport("pyscf.symm")
    mol.atom = molecule
    mol.max_memory = 1000  # MB
    mol.symmetry = true
    mol.charge = charge
    mol.spin = spin
    mol.basis = basis_set
    mol.build()
    gto=pyimport("pyscf.gto")
    println("symmetry")
    println(mol.topgroup)
    mcscf=pyimport("pyscf.mcscf")
    #SCF 

    mf = PySCF.scf.RHF(mol).run(conv_tol=1e-8, verbose=4)
    enu = mf.energy_nuc()
    C=mf.mo_coeff
    println("MO Energies")
    display(mf.mo_energy)
    if mol.symmetry
        mo = symm.symmetrize_orb(mol, mf.mo_coeff)
        osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
        for i in 1:length(osym)
            println(rpad(i, 4), " ", rpad(osym[i], 8), " ", rpad(mf.mo_energy[i], 16))
        end
    end
    
    n_orb =gto.nao_nr(mol)
    n_b, n_a = mol.nelec
    nel = n_a + n_b
    #cas 
    if cas
        cas_norb = cas_nstop - cas_nstart+1
        @assert cas_nstart !== nothing
        @assert cas_nstop !== nothing
        @assert cas_nel !== nothing
    else
        cas_nstart = 0
        cas_nstop = n_orb
        cas_nel = nel
    end
    pyscflo = pyimport("pyscf.lo")
    ## AO to MO Transformation: orb_basis or scf
    if orb_basis == "scf"
        println("\nUsing Canonical Hartree Fock orbitals...\n")
        C = copy(mf.mo_coeff)
        println("C shape")
        println(size(C))
    
    elseif orb_basis == "lowdin"
        @assert !cas
        S = mol.intor("int1e_ovlp_sph")
        println("Using lowdin orthogonalized orbitals")
    
        C = lowdin(S)
        # end
        
    elseif orb_basis == "boys"
        PySCF.lib.num_threads(1)  # with degenerate states and multiple processors there can be issues
        cl_c = mf.mo_coeff[:, 1:cas_nstart]
        cl_a = pyscflo.Boys(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, cas_nstop:end]
        C = hcat(cl_c, cl_a, cl_v)
    
    elseif orb_basis == "boys2"
        PySCF.lib.num_threads(1)  # with degenerate states and multiple processors there can be issues
        cl_c = mf.mo_coeff[:, 1:loc_nstart]
        cl_a = pyscflo.Boys(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, loc_nstop:end]
        C = hcat(cl_c, cl_a, cl_v)
    
    elseif orb_basis == "PM"
        PySCF.lib.num_threads(1) # with degenerate states and multiple processors there can be issues
        cl_c = mf.mo_coeff[:, 1:cas_nstart]
        cl_a = pyscflo.PM(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, cas_nstop:end]
        C = hcat(cl_c, cl_a, cl_v)
    
    elseif orb_basis == "PM2"
        PySCF.lib.num_threads(1)  # with degenerate states and multiple processors there can be issues
        cl_c = mf.mo_coeff[:, 1:loc_nstart]
        cl_a = pyscflo.PM(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, loc_nstop:end]
        C = hcat(cl_c, cl_a, cl_v)
    
    elseif orb_basis == "ER"
        PySCF.lib.num_threads(1) # with degenerate states and multiple processors there can be issues
        cl_c = mf.mo_coeff[:, 1:cas_nstart]
        cl_a = py"lo.PM"(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, cas_nstop:end]
        C = hcat(cl_c, cl_a, cl_v)
    
    elseif orb_basis == "ER2"
        PySCF.lib.num_threads(1)  # with degenerate states and multiple processors there can be issues
        cl_c = mf.mo_coeff[:, 1:loc_nstart]
        cl_a = pyscflo.ER(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
        cl_v = mf.mo_coeff[:, loc_nstop:end]
        C = hcat(cl_c, cl_a, cl_v)
    end
    
    if cas
        cas_nstart = cas_nstart === nothing ? nothing : cas_nstart
        cas_nstop = cas_nstop === nothing ? nothing : cas_nstop
        cas_nel = cas_nel === nothing ? nothing : cas_nel

        @assert cas_nstart !== nothing
        @assert cas_nstop !== nothing
        @assert cas_nel !== nothing
        
        
        cas_norb=(cas_nstop-cas_nstart)+1
        mycas = mcscf.CASSCF(mf, cas_norb, cas_nel)
        h1e_cas, ecore = mycas.get_h1eff(mo_coeff = C)  # core orbs to form ecore and eff
        h2e_cas = ao2mo.kernel(mol, C[:,cas_nstart:cas_nstop], aosym="s4", compact=false)
        h2e_cas = reshape(h2e_cas,(cas_norb,cas_norb,cas_norb,cas_norb)) 
        println(h1e_cas)
        #return h1e_cas,h2e_cas,ecore,C,mol,mf
        helper.h = h1e_cas
        helper.g = h2e_cas
        helper.ecore = ecore
        helper.mf = mf
        helper.mol = mol
        helper.C = deepcopy(C[:,cas_nstart:cas_nstop])
        J,K = mf.get_jk()
        helper.J = helper.C' *J *helper.C
        helper.K = helper.C' *J *helper.C
        helper.S=mol.intor("int1e_ovlp_sph")
        if false
            helper.C = C
            helper.S=
            h = C' * mf.get_hcore() *C
            g_= ao2mo.kernel(mol, C, aosym="s4",compact=false)[:] 
            display(g_)
            g = reshape(g_, (n_orb,n_orb,n_orb,n_orb)) 
            constANT,heff = get_eff_for_casci(cas_nstart,cas_nstop,h,g_)
            println(constANT)
            println(heff)
            helper.n_orb = n_orb
            idx = cas_nstart:cas_nstop-1
            h = h[:, idx]
            h = h[idx, :]
            g = g[:, :, :, idx]
            g = g[:, :, idx, :]
            g = g[:, idx, :, :]
            g = g[idx, :, :, :]

            helper.ecore = constANT
            helper.h = h + heff
            helper.g = g
            helper.S=mol.intor("int1e_ovlp_sph")
            display(g)
        end
    else cas=false
        helper.C = C
        h = C' * mf.get_hcore() *C
        g_= ao2mo.kernel(mol, C, aosym="s4",compact=false)[:] 
        g = reshape(g_, (n_orb,n_orb,n_orb,n_orb)) 
        helper.h=h
        helper.g=g
        helper.mf = mf
        helper.mol = mol
        helper.ecore = enu
        helper.S = PySCF.scf.RHF(mol).get_ovlp()
        J, K = mf.get_jk()
        helper.J = helper.C' * J * helper.C
        helper.K = helper.C' * K * helper.C
        
    end

    
end
function get_eff_for_casci(n_start, n_stop, h, g)
    constant = 0.0
    for i in 1:n_start
        constant += 2 * h[i, i]
        for j in 1:n_start
            constant += 2 * g[i, i, j, j] - g[i, j, i, j]
        end
    end

    eff = zeros(n_stop - n_start, n_stop - n_start)

    for l in n_start:n_stop-1
        L = l - n_start + 1
        for m in n_start:n_stop-1
            M = m - n_start + 1
            for j in 1:n_start
                eff[L, M] += 2 * g[l, m, j, j] - g[l, j, j, m]
            end
        end
    end

    return constant, eff
end

helper = PyscfHelper()
molecule = "H 0.0 0.0 0.0; H 0.0 0.0 0.74"
charge = 0
spin = 0
basis_set = "sto-3g"
init(helper, molecule, charge, spin, basis_set, orb_basis="boys", cas=true,cas_nstart=1, cas_nstop=2,cas_nel=2)

# Access the results
println("Total SCF energy: ", helper.Escf)
println("Orbital coefficient matrix: ")
println(helper.C)
println("One-electron integrals: ")
println(helper.h)
println("Two-electron integrals: ")
println(helper.g)
println("Overlap matrix: ")
println(helper.S)
println("Coulomb matrix: ")
println(helper.J)
println("Exchange matrix: ")
println(helper.K)
function run_fci_pyscf(h, g, nelec, ecore=0, nroots=1)
    # FCI
    cisolver = PySCF.fci.direct_spin1.FCI()
    efci, ci = cisolver.kernel(h, g, size(h, 2), nelec, ecore=ecore, nroots=nroots, verbose=100)
    fci_dim = size(ci, 1) * size(ci, 2)
    d1 = cisolver.make_rdm1(ci, size(h, 2), nelec)
    println(d1)
    println(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
    println("FCI %10.8f" % efci)
    
    return efci, fci_dim
end
function run_hci_pyscf(h, g, nelec, ecore=0, select_cutoff=5e-4, ci_cutoff=5e-4)
    # Heat-bath CI
    cisolver = PySCF.hci.SCI()
    cisolver.select_cutoff = select_cutoff
    cisolver.ci_coeff_cutoff = ci_cutoff
    ehci, civec = cisolver.kernel(h, g, size(h, 2), nelec, ecore=ecore, verbose=4)
    hci_dim = size(civec[1], 1)
    println(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))
    println("HCI %10.8f" % ehci)
    
    return ehci, hci_dim
end
function reorder_integrals(idx, h, g)
    h = h[:, idx]
    h = h[idx, :]

    g = g[:, :, :, idx]
    g = g[:, :, idx, :]
    g = g[:, idx, :, :]
    g = g[idx, :, :, :]
    
    return h, g
end

using SparseArrays
using SuiteSparseGraphBLAS

function e1_order(h, cut_off)
    """This function takes a matrix h and a cutoff value cut_off. It calculates the absolute values of h,
     sets values below the cutoff to 0, and then performs the Reverse Cuthill-McKee (RCM) 
     reordering on the resulting matrix. The function returns the reordering index idx."""
    hnew = abs.(h)
    hnew[hnew .< cut_off] .= 0
    fill!(Diagonal(hnew), 0)
    
    hnew_sparse = sparse(hnew)
    idx = reverse_cuthill_mckee(hnew_sparse)
    idx = idx .+ 1
    
    hnew = hnew[:, idx]
    hnew = hnew[idx, :]
    
    println("New order:")
    println(hnew)
    
    return idx
end
using SparseArrays
using SuiteSparseGraphBLAS

function ordering(pmol, cas, cas_nstart, cas_nstop, loc_nstart, loc_nstop, ordering="hcore")
    """This function takes various parameters (pmol, cas, cas_nstart, cas_nstop, loc_nstart, loc_nstop, ordering) and performs ordering based
     on the specified ordering method (default is "hcore"). It initializes loc_range and out_range arrays based on the given start and stop indices. 
     It then deep copies the pmol.h matrix and applies the ordering based on the chosen method"""
    loc_range = (loc_nstart - cas_nstart):(loc_nstop - cas_nstart - 1)
    out_range = (loc_nstop - cas_nstart):(cas_nstop - cas_nstart - 1)
    println(loc_range)
    println(out_range)
    
    h = deepcopy(pmol.h)
    println(h)
    
    if ordering == "hcore"
        println("Bonding Active Space")
        hl = h[:, loc_range]
        hl = hl[loc_range, :]
        println(hl)
        idl = e1_order(hl,1e-2)
        
        ho = h[:, out_range]
        ho = ho[out_range, :]
        println("Virtual Active Space")
        ido = e1_order(ho, 1e-2)
        
        idl = idl .+ 1
        ido = ido .+ loc_nstop - cas_nstart
    end
    
    println(idl)
    println(ido)
    idx = vcat(idl, ido)
    println(idx)
    
    return idx
end
mo_mapping=pyimport("pyscf.tools.mo_mapping")

function mo_comps(orb, mol, C)
    s_pop = mo_mapping.mo_comps(orb, mol, C)
    return convert(Array{Float64, 1}, s_pop)
end
function ordering_diatomics(mol, C)
    # DZ basis diatomics reordering with frozen 1s

    orb_type = ["s", "pz", "dz", "px", "dxz", "py", "dyz", "dx2-y2", "dxy"]
    ref = zeros(Int, size(C, 2))

    # Find dimension of each space
    dim_orb = []
    for orb in orb_type
        println("Orb type: ", orb)
        idx = 0
        for label in mol.ao_labels()
            if occursin(orb, label)
                idx += 1
            end
        end

        # frozen 1s orbitals
        if orb == "s"
            idx -= 2
        end
        push!(dim_orb, idx)
        println(idx)
    end

    new_idx = Int[]
    # Find orbitals corresponding to each orb space
    for (i, orb) in enumerate(orb_type)
        println("Orbital type: ", orb)
        s_pop = mo_comps(orb, mol, C)
        ref .+= s_pop
        cas_list = sortperm(s_pop, rev=true)[end-dim_orb[i]+1:end]
        println("cas_list: ", cas_list)
        append!(new_idx, cas_list)
    end

    ao_labels = ao_labels(mol)
    println(ref)
    println(new_idx)
    for label in ao_labels
        println(label)
    end

    return new_idx
end
