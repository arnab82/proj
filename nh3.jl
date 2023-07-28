include("clustering.jl")
include("cmf.jl")
include("pyscfhelper.jl")
using PyCall
pyscf=pyimport("pyscf")
pyscf.lib.num_threads(1)

helper = PyscfHelper()
molecule = "N 1.0 1.0 1.0; H 0.6 0.6 0.0; H 0.0 0.6 0.6; H 0.6 0.0 0.6"
charge = 0
spin = 0
basis_set = "sto-3g"
cas = true
cas_nstart = 1
cas_nstop =  8
loc_start = 1
loc_stop = 8
cas_nel = 8
init(helper, molecule, charge, spin, basis_set, orb_basis="boys", cas=cas,cas_nstart=cas_nstart, cas_nstop=cas_nstop,cas_nel=cas_nel)
h=helper.h
g=helper.g
s=helper.S
j=helper.J
k=helper.K
c=helper.C
ecore=helper.ecore
blocks = [[1,2,3,4],[5,6,7,8]]
init_fspace = ((2, 2), (2, 2))
block_nel = [[2,2], [2,2]]
nelec = 8
println(init_fspace[1][1])
println(typeof(blocks[1]))

efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore)
println(efci)

idx = e1_order(h,3e-1)
h,g = reorder_integrals(idx,h,g)
maximum_iteration=50
Ecmf = run_cmf(h,g,blocks,block_nel,ecore,maximum_iteration)
miter=0
Edps = run_cmf(h,g,blocks,block_nel,ecore,miter)
Escf = helper.mf.e_tot