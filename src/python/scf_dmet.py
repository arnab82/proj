import pyscf
import pyscf.tools
from scipy import linalg
import sys
#from pyscf import gto 
import numpy as np
def input_file():
    inFile = sys.argv[1]
    with open(inFile,'r') as i:
        content = i.readlines()
    input_file =[]
    for line in content:
        v_line=line.strip()
        if len(v_line)>0:
            input_file.append(v_line.split())
    Level_of_theory = input_file[0][0]
    basis_set = input_file[0][1]
    unit = input_file[0][2]
    sym= input_file[0][3]
    conv= input_file[0][4]
    grad_tol= input_file[0][5]
    max_cycle= input_file[0][6]
    initial_guess=input_file[0][7]
    fragno_given=input_file[0][8]
    charge_, spin_= input_file[1]
    for i in range(2):
        input_file.pop(0)
    geom_file = input_file
    Atoms = []
    for i in range(len(geom_file)):
        Atoms.append(geom_file[i][0])
    #print(Atoms)
    geom_raw = geom_file
    for i in range(len(geom_file)):
        geom_raw[i].pop(0)
    geom = ''
    atomline = ''
    for i in range(len(geom_raw)):
        atomline += Atoms[i]+" "
        for j in range(len(geom_raw[i])):
            if j!=(len(geom_raw[i])-1):
                atomline += geom_raw[i][j]+" "
            else:
                atomline += geom_raw[i][j]
		
        if (i == len(geom_raw)-1):
            geom += atomline +""
        else:
            geom += atomline +";"
        atomline = ''
    print(geom)
    print(type(geom))
    print(basis_set)
    print(Level_of_theory)
    return spin_,charge_,geom,unit,basis_set,sym,Level_of_theory,conv,grad_tol,max_cycle,initial_guess,fragno_given
spin_,charge_,geom,unit,basis_set,sym,Level_of_theory,conv,grad_tol,max_cycle,initial_guess,fragno_given=input_file()
print(spin_,charge_,geom,unit,basis_set,sym,Level_of_theory,conv,grad_tol,max_cycle,initial_guess)
'''molecule = """
 Fe 1.67785607 0.00052233 0.06475932
 Fe -1.67785607 -0.00052233 0.06475932
 O 0.00000000 0.00000000 -0.47099074
 Cl 1.87002704 -1.09796437 1.99091682
 Cl 2.93244917 -0.98210488 -1.47467288
 Cl 2.37160936 2.07954091 -0.50446591
 Cl -1.87002704 1.09796437 1.99091682
 Cl -2.93244917 0.98210488 -1.47467288
 Cl -2.37160936 -2.07954091 -0.50446591
"""

basis = "def2-svp"'''
pymol = pyscf.gto.Mole(
        atom    =   geom,
        symmetry=   True,
        spin    =   int(spin_), # number of unpaired electrons
        charge  =   int(charge_),
        basis   =   basis_set)
pymol.build()
print("symmetry: ",pymol.topgroup)
if Level_of_theory=="ROHF":
    mf = pyscf.scf.ROHF(pymol)
elif Level_of_theory=="RHF":
    mf = pyscf.scf.RHF(pymol)
elif Level_of_theory=="UHF_X2C":
    mf = pyscf.scf.UHF(pymol).x2c()
else:
    mf=pyscf.scf.UHF(pymol)
mf.verbose = 4
mf.conv_tol = float(conv)
mf.conv_tol_grad = float(grad_tol)
mf.chkfile = "scf.fchk"
mf.init_guess = initial_guess
mf.run(max_cycle=int(max_cycle))

print(" Hartree-Fock Energy: %12.8f" % mf.e_tot)
# mf.analyze()

F = mf.get_fock()

import numpy as np
import scipy
import copy as cp
import math

def get_frag_bath(Pin, frag, S, thresh=1e-7, verbose=2):
    #argument: takes density matrix, fragment, overlap matrix
    #return: mo coefficients
    print(" Frag: ", frag)
    X = scipy.linalg.sqrtm(S)
    Xinv = scipy.linalg.inv(X)

    Nbas = S.shape[0]
    Cfrag = Xinv@np.eye(Nbas)[:,frag]
    

    P = X@Pin@X.T

    nfrag = np.trace(P[frag,:][:,frag])
    P[frag,:] = 0
    P[:,frag] = 0
    bath_idx = []
    env_idx = []
    e,U = np.linalg.eigh(P)
    nbath = 0.0
    for nidx,ni in enumerate(e):
        if math.isclose(ni, 1, abs_tol=thresh):
            env_idx.append(nidx)
        elif thresh < ni < 1-thresh:
            if verbose > 1:
                print(" eigvalue: %12.8f" % ni)
            bath_idx.append(nidx)
            nbath += ni
        
            
    print(" # Electrons frag: %12.8f bath: %12.8f total: %12.8f" %(nfrag, nbath, nfrag+nbath))
    Cenv = Xinv@U[:,env_idx]
    Cbath = Xinv@U[:,bath_idx]
    C = np.hstack((Cfrag, Cbath))
    
    # Get virtual orbitals (these are just the orthogonal complement of the env and frag/bath
    Q = np.eye(C.shape[0]) - X@C@C.T@X - X@Cenv@Cenv.T@X
    e,U = np.linalg.eigh(Q)
    vir_idx = []
    for nidx,ni in enumerate(e):
        if math.isclose(ni, 1, abs_tol=thresh):
            vir_idx.append(nidx)
    Cvir = Xinv@U[:,vir_idx]

    assert(Cenv.shape[1] + Cvir.shape[1] + C.shape[1] == Nbas)

          
    # print(C.T@S@C)
    return (Cenv, C, Cvir)

def gram_schmidt(frags, S, thresh=1e-8):
    # |v'> = (1-sum_i |i><i|) |v>
    #      = |v> - sum_i |i><i|v>
    Nbas = S.shape[1]
    seen = []
    out = []
    seen = np.zeros((Nbas, 0))

    for f in frags:
        outf = np.zeros((Nbas, 0))
        # grab each orbital
        for fi in range(f.shape[1]):
            v = f[:,fi]
            v.shape = (Nbas, 1)

            # Compare to previous orbitals
            for fj in range(seen.shape[1]):
                j = seen[:,fj]
                j.shape = (Nbas, 1)
                ovlp = (j.T @ S @ v)[0]
                v = v - j * ovlp

                norm = np.sqrt((v.T @ S @ v)[0])
                if norm < thresh:
                    print(" Warning: small norm in GS: ", norm)
                v = v/norm
            
            outf = np.hstack((outf, v))
            seen = np.hstack((seen, v))
        out.append(outf)
    return out

def sym_ortho(frags, S, thresh=1e-8):
    Nbas = S.shape[1]
    
    inds = []
    Cnonorth = np.hstack(frags)
    shift = 0
    for f in frags:
        inds.append(list(range(shift, shift+f.shape[1])))
        shift += f.shape[1]
        
    
    Smo = Cnonorth.T @ S @ Cnonorth
    X = np.linalg.inv(linalg.sqrtm(Smo))
    # print(Cnonorth.shape, X.shape)
    Corth = Cnonorth @ X
    
    frags2 = []
    for f in inds:
        frags2.append(Corth[:,f])
    return frags2
def frags3():
    full = []
    frag1 = []
    frag2 = []
    frag3 = []
    for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
        if ao[0] == 0:
            if ao[2] in ("3d"):
                frag1.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 1:
            if ao[2] in ("3d"):
                frag3.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 2:
            if ao[2] in ("2p"):
                frag2.append(ao_idx)
                full.append(ao_idx)

    frags = [frag1, frag2, frag3]
    P = mf.make_rdm1()
    Pa = P[0,:,:]
    Pb = P[1,:,:]

    C = mf.mo_coeff
    S = mf.get_ovlp()

    (Cenv, Cact, Cvir) = get_frag_bath(Pa, full, S)
    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)

    CC = np.hstack((Cenv, Cact, Cvir))
    # Since we used Pa for the orbital partitioning, we will have an integer number of alpha electrons,
    # but not beta. however, we can determine number of alpha because the env is closed shell.
    n_tot_b = mf.nelec[1]
    na_act = np.trace(Cact.T @ S @ Pa @ S @ Cact)
    na_env = np.trace(Cenv.T @ S @ Pa @ S @ Cenv)
    na_vir = np.trace(Cvir.T @ S @ Pa @ S @ Cvir)
    nb_env = na_env
    nb_act = n_tot_b - nb_env
    nb_vir = na_vir
    print(" # electrons: %12s %12s" %("α", "β"))
    print("         Env: %12.8f %12.8f" %(na_env, nb_env))
    print("         Act: %12.8f %12.8f" %(na_act, nb_act))
    print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))

    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)


    Pa = Cact @ Cact.T @ S @ Pa @ S @ Cact @ Cact.T

    print(" Now localize")
    (Ce1, Cf1, Cv1) = get_frag_bath(Pa, frag1, S)
    (Ce2, Cf2, Cv2) = get_frag_bath(Pa, frag2, S)
    (Ce3, Cf3, Cv3) = get_frag_bath(Pa, frag3, S)


    # frag_orbs = gram_schmidt((Cf1, Cf2, Cf3), S)
    frag_orbs = sym_ortho((Cf1, Cf2, Cf3), S)
    return frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C

def frags3_():
    full = []
    frag1 = []
    frag2 = []
    frag3 = []
    for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
        if ao[0] == 0:
            if ao[2] in ("3d"):
                frag1.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 1:
            if ao[2] in ("3d"):
                frag3.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 2:
            if ao[2] in ("2p"):
                frag2.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 3:
            if ao[2] in ("2p"):
                frag2.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 4:
            if ao[2] in ("2p"):
                frag2.append(ao_idx)
                full.append(ao_idx)

    frags = [frag1, frag2, frag3]
    P = mf.make_rdm1()
    Pa = P[0,:,:]
    Pb = P[1,:,:]

    C = mf.mo_coeff
    S = mf.get_ovlp()

    (Cenv, Cact, Cvir) = get_frag_bath(Pa, full, S)
    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)

    CC = np.hstack((Cenv, Cact, Cvir))
    # Since we used Pa for the orbital partitioning, we will have an integer number of alpha electrons,
    # but not beta. however, we can determine number of alpha because the env is closed shell.
    n_tot_b = mf.nelec[1]
    na_act = np.trace(Cact.T @ S @ Pa @ S @ Cact)
    na_env = np.trace(Cenv.T @ S @ Pa @ S @ Cenv)
    na_vir = np.trace(Cvir.T @ S @ Pa @ S @ Cvir)
    nb_env = na_env
    nb_act = n_tot_b - nb_env
    nb_vir = na_vir
    print(" # electrons: %12s %12s" %("α", "β"))
    print("         Env: %12.8f %12.8f" %(na_env, nb_env))
    print("         Act: %12.8f %12.8f" %(na_act, nb_act))
    print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))

    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)


    Pa = Cact @ Cact.T @ S @ Pa @ S @ Cact @ Cact.T

    print(" Now localize")
    (Ce1, Cf1, Cv1) = get_frag_bath(Pa, frag1, S)
    (Ce2, Cf2, Cv2) = get_frag_bath(Pa, frag2, S)
    (Ce3, Cf3, Cv3) = get_frag_bath(Pa, frag3, S)


    # frag_orbs = gram_schmidt((Cf1, Cf2, Cf3), S)
    frag_orbs = sym_ortho((Cf1, Cf2, Cf3), S)
    return frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C
# Find AO's corresponding to atoms 
# print(mf.mol.aoslice_by_atom())
# print(mf.mol.ao_labels(fmt=False, base=0))
def frags2():
    full = []
    frag1 = []
    frag2 = []
    for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
        if ao[0] == 0:
            if ao[2] in ("3d"):
                frag1.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 1:
            if ao[2] in ("3d"):
                frag2.append(ao_idx)
                full.append(ao_idx)

    frags = [frag1, frag2]
    P = mf.make_rdm1()
    Pa = P[0,:,:]
    Pb = P[1,:,:]

    C = mf.mo_coeff
    S = mf.get_ovlp()

    (Cenv, Cact, Cvir) = get_frag_bath(Pa, full, S)
    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)

    CC = np.hstack((Cenv, Cact, Cvir))
    # Since we used Pa for the orbital partitioning, we will have an integer number of alpha electrons,
    # but not beta. however, we can determine number of alpha because the env is closed shell.
    n_tot_b = mf.nelec[1]
    na_act = np.trace(Cact.T @ S @ Pa @ S @ Cact)
    na_env = np.trace(Cenv.T @ S @ Pa @ S @ Cenv)
    na_vir = np.trace(Cvir.T @ S @ Pa @ S @ Cvir)
    nb_env = na_env
    nb_act = n_tot_b - nb_env
    nb_vir = na_vir
    print(" # electrons: %12s %12s" %("α", "β"))
    print("         Env: %12.8f %12.8f" %(na_env, nb_env))
    print("         Act: %12.8f %12.8f" %(na_act, nb_act))
    print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))

    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)


    Pa = Cact @ Cact.T @ S @ Pa @ S @ Cact @ Cact.T

    print(" Now localize")
    (Ce1, Cf1, Cv1) = get_frag_bath(Pa, frag1, S)
    (Ce2, Cf2, Cv2) = get_frag_bath(Pa, frag2, S)


    # frag_orbs = gram_schmidt((Cf1, Cf2, Cf3), S)
    frag_orbs = sym_ortho((Cf1, Cf2), S)
    return frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C
def frags4():
    full = []
    frag1 = []
    frag2 = []
    frag3 = []
    frag4 =[]
    for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
        if ao[0] == 0:
            if ao[2] in ("2s", "2p"):
                frag1.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 1:
            if ao[2] in ("2s", "2p"):
                frag3.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 2:
            if ao[2] in ("2s", "2p"):
                frag2.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 3:
            if ao[2] in ("2s", "2p"):
                frag4.append(ao_idx)
                full.append(ao_idx)

    frags = [frag1, frag2, frag3,frag4]
    P = mf.make_rdm1()
    Pa = P[0,:,:]
    Pb = P[1,:,:]

    C = mf.mo_coeff
    S = mf.get_ovlp()

    (Cenv, Cact, Cvir) = get_frag_bath(Pa, full, S)
    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)

    CC = np.hstack((Cenv, Cact, Cvir))
    # Since we used Pa for the orbital partitioning, we will have an integer number of alpha electrons,
    # but not beta. however, we can determine number of alpha because the env is closed shell.
    n_tot_b = mf.nelec[1]
    na_act = np.trace(Cact.T @ S @ Pa @ S @ Cact)
    na_env = np.trace(Cenv.T @ S @ Pa @ S @ Cenv)
    na_vir = np.trace(Cvir.T @ S @ Pa @ S @ Cvir)
    nb_env = na_env
    nb_act = n_tot_b - nb_env
    nb_vir = na_vir
    print(" # electrons: %12s %12s" %("α", "β"))
    print("         Env: %12.8f %12.8f" %(na_env, nb_env))
    print("         Act: %12.8f %12.8f" %(na_act, nb_act))
    print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))

    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)


    Pa = Cact @ Cact.T @ S @ Pa @ S @ Cact @ Cact.T

    print(" Now localize")
    (Ce1, Cf1, Cv1) = get_frag_bath(Pa, frag1, S)
    (Ce2, Cf2, Cv2) = get_frag_bath(Pa, frag2, S)
    (Ce3, Cf3, Cv3) = get_frag_bath(Pa, frag3, S)
    (Ce4, Cf4, Cv4) = get_frag_bath(Pa, frag4, S)


    # frag_orbs = gram_schmidt((Cf1, Cf2, Cf3), S)
    frag_orbs = sym_ortho((Cf1, Cf2, Cf3,Cf4), S)
    return frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C
def frags5():
    full = []
    frag1 = []
    frag2 = []
    frag3 = []
    frag4 =[]
    frag5 =[]
    for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
        if ao[0] == 0:
            if ao[2] in ("3d"):
                frag1.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 1:
            if ao[2] in ("3d"):
                frag3.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 2:
            if ao[2] in ("2p"):
                frag2.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 3:
            if ao[2] in ("2p"):
                frag4.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 4:
            if ao[2] in ("2p"):
                frag4.append(ao_idx)
                full.append(ao_idx)
    frags = [frag1, frag2, frag3,frag4,frag5]
    P = mf.make_rdm1()
    Pa = P[0,:,:]
    Pb = P[1,:,:]

    C = mf.mo_coeff
    S = mf.get_ovlp()

    (Cenv, Cact, Cvir) = get_frag_bath(Pa, full, S)
    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)

    CC = np.hstack((Cenv, Cact, Cvir))
    # Since we used Pa for the orbital partitioning, we will have an integer number of alpha electrons,
    # but not beta. however, we can determine number of alpha because the env is closed shell.
    n_tot_b = mf.nelec[1]
    na_act = np.trace(Cact.T @ S @ Pa @ S @ Cact)
    na_env = np.trace(Cenv.T @ S @ Pa @ S @ Cenv)
    na_vir = np.trace(Cvir.T @ S @ Pa @ S @ Cvir)
    nb_env = na_env
    nb_act = n_tot_b - nb_env
    nb_vir = na_vir
    print(" # electrons: %12s %12s" %("α", "β"))
    print("         Env: %12.8f %12.8f" %(na_env, nb_env))
    print("         Act: %12.8f %12.8f" %(na_act, nb_act))
    print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))

    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)


    Pa = Cact @ Cact.T @ S @ Pa @ S @ Cact @ Cact.T

    print(" Now localize")
    (Ce1, Cf1, Cv1) = get_frag_bath(Pa, frag1, S)
    (Ce2, Cf2, Cv2) = get_frag_bath(Pa, frag2, S)
    (Ce3, Cf3, Cv3) = get_frag_bath(Pa, frag3, S)
    (Ce4, Cf4, Cv4) = get_frag_bath(Pa, frag4, S)
    (Ce5, Cf5, Cv5) = get_frag_bath(Pa, frag5, S)


    # frag_orbs = gram_schmidt((Cf1, Cf2, Cf3), S)
    frag_orbs = sym_ortho((Cf1, Cf2, Cf3,Cf4,Cf5), S)
    return frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C


def frags4_():
    full = []
    frag1 = []
    frag2 = []
    frag3 = []
    frag4 = []
    for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
        if ao[0] == 0:
            if ao[2] in ("3s", "3p"):
                frag1.append(ao_idx)
                full.append(ao_idx)
            elif ao[2] in ("3d",):
                frag2.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 1:
            if ao[2] in ("3s", "3p"):
                frag3.append(ao_idx)
                full.append(ao_idx)
            if ao[2] in ("3d",):
                frag4.append(ao_idx)
                full.append(ao_idx)


    frags = [frag1, frag2, frag3, frag4]
    P = mf.make_rdm1()
    Pa = P[0,:,:]
    Pb = P[1,:,:]

    C = mf.mo_coeff
    S = mf.get_ovlp()

    (Cenv, Cact, Cvir) = get_frag_bath(Pa, full, S)
    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)

    CC = np.hstack((Cenv, Cact, Cvir))

    # Since we used Pa for the orbital partitioning, we will have an integer number of alpha electrons,
    # but not beta. however, we can determine number of alpha because the env is closed shell.
    n_tot_b = mf.nelec[1]
    na_act = np.trace(Cact.T @ S @ Pa @ S @ Cact)
    na_env = np.trace(Cenv.T @ S @ Pa @ S @ Cenv)
    na_vir = np.trace(Cvir.T @ S @ Pa @ S @ Cvir)
    nb_env = na_env
    nb_act = n_tot_b - nb_env
    nb_vir = na_vir
    print(" # electrons: %12s %12s" %("α", "β"))
    print("         Env: %12.8f %12.8f" %(na_env, nb_env))
    print("         Act: %12.8f %12.8f" %(na_act, nb_act))
    print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))

    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)


    Pa = Cact @ Cact.T @ S @ Pa @ S @ Cact @ Cact.T

    (Ce1, Cf1, Cv1) = get_frag_bath(Pa, frag1, S)
    (Ce2, Cf2, Cv2) = get_frag_bath(Pa, frag2, S)
    (Ce3, Cf3, Cv3) = get_frag_bath(Pa, frag3, S)
    (Ce4, Cf4, Cv4) = get_frag_bath(Pa, frag4, S)

    frag_orbs = gram_schmidt((Cf1, Cf2, Cf3, Cf4), S)
    return frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C
def frags5_():
    full = []
    frag1 = []
    frag2 = []
    frag3 = []
    frag4 = []
    frag5=[]
    for ao_idx,ao in enumerate(mf.mol.ao_labels(fmt=False)):
        if ao[0] == 0:
            if ao[2] in ("3s", "3p"):
                frag1.append(ao_idx)
                full.append(ao_idx)
            elif ao[2] in ("3d",):
                frag2.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 1:
            if ao[2] in ("3s", "3p"):
                frag3.append(ao_idx)
                full.append(ao_idx)
            if ao[2] in ("3d",):
                frag4.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 2:
            if ao[2] in ("2p"):
                frag5.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 3:
            if ao[2] in ("2p"):
                frag5.append(ao_idx)
                full.append(ao_idx)
        elif ao[0] == 4:
            if ao[2] in ("2p"):
                frag5.append(ao_idx)
                full.append(ao_idx)


    frags = [frag1, frag2, frag3, frag4,frag5]
    P = mf.make_rdm1()
    Pa = P[0,:,:]
    Pb = P[1,:,:]

    C = mf.mo_coeff
    S = mf.get_ovlp()

    (Cenv, Cact, Cvir) = get_frag_bath(Pa, full, S)
    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)

    CC = np.hstack((Cenv, Cact, Cvir))

    # Since we used Pa for the orbital partitioning, we will have an integer number of alpha electrons,
    # but not beta. however, we can determine number of alpha because the env is closed shell.
    n_tot_b = mf.nelec[1]
    na_act = np.trace(Cact.T @ S @ Pa @ S @ Cact)
    na_env = np.trace(Cenv.T @ S @ Pa @ S @ Cenv)
    na_vir = np.trace(Cvir.T @ S @ Pa @ S @ Cvir)
    nb_env = na_env
    nb_act = n_tot_b - nb_env
    nb_vir = na_vir
    print(" # electrons: %12s %12s" %("α", "β"))
    print("         Env: %12.8f %12.8f" %(na_env, nb_env))
    print("         Act: %12.8f %12.8f" %(na_act, nb_act))
    print("         Vir: %12.8f %12.8f" %(na_vir, nb_vir))

    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Cact)


    Pa = Cact @ Cact.T @ S @ Pa @ S @ Cact @ Cact.T

    (Ce1, Cf1, Cv1) = get_frag_bath(Pa, frag1, S)
    (Ce2, Cf2, Cv2) = get_frag_bath(Pa, frag2, S)
    (Ce3, Cf3, Cv3) = get_frag_bath(Pa, frag3, S)
    (Ce4, Cf4, Cv4) = get_frag_bath(Pa, frag4, S)
    (Ce5, Cf5, Cv5) = get_frag_bath(Pa, frag5, S)

    frag_orbs = gram_schmidt((Cf1, Cf2, Cf3, Cf4,Cf5), S)
    return frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C


def cluster_making():
    if fragno_given=="fragno_3":
        frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C=frags3()
    elif fragno_given=="fragno_2":
        frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C=frags2()
    elif fragno_given=="fragno_4_":
        frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C=frags4_()
    elif fragno_given=="fragno_3_":
        frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C=frags3_()
    elif fragno_given=="fragno_5_":
        frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C=frags5_()
    elif fragno_given=="fragno_4":
        frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C=frags4()
    elif fragno_given=="fragno_5":
        frag_orbs,Cenv,Cvir,Cact,Pa,Pb,S,frags,C=frags5()
    Nbas = S.shape[1]
    Ctot = np.zeros((Nbas,0))

    clusters = []
    init_fspace = []

    ci_shift = 0
    for fi,f in enumerate(frag_orbs):
        assert np.linalg.norm(f.T @ S @ Cenv) < 1e-12 
        assert np.linalg.norm(f.T @ S @ Cvir) < 1e-12 

        # Canonicalize
        focki = f.T @ F @ f
        ei,vi = np.linalg.eigh(focki)
        #for ei_idx,e in enumerate(ei):
            #print(" %4i %12.8f"%(ei_idx, e))
        
        # f = f@vi
        
        Ctot = np.hstack((Ctot, f))
        Paf = f.T @ S @ Pa @ S @ f
        Pbf = f.T @ S @ Pb @ S @ f
        
        na = np.trace(Paf)
        nb = np.trace(Pbf)
        
        clusters.append(list(range(ci_shift, ci_shift+f.shape[1])))
        ci_shift += f.shape[1]
        init_fspace.append((int(np.round(na)), int(np.round(nb))))
        print(" Fragment: %4i %s" %(fi,frags[fi]))
        print("    # α electrons: %12.8f" % na)
        print("    # β electrons: %12.8f" % nb)

        # pyscf.tools.molden.from_mo(mf.mol, "C_frag%2i.molden"%fi, f);

        
    pyscf.tools.molden.from_mo(mf.mol, "C_act.molden", Ctot);
    print(clusters)
    print(init_fspace)

    return Ctot,clusters,init_fspace,Cenv,Cvir
Ctot,clusters,init_fspace,Cenv,Cvir=cluster_making()

def integral_making():
    d1_embed = 2 * Cenv @ Cenv.T

    h0 = pyscf.gto.mole.energy_nuc(mf.mol)
    h  = pyscf.scf.hf.get_hcore(mf.mol)
    j, k = pyscf.scf.hf.get_jk(mf.mol, d1_embed, hermi=1)
    h0 += np.trace(d1_embed @ ( h + .5*j - .25*k))

    h = Ctot.T @ h @ Ctot;
    j = Ctot.T @ j @ Ctot;
    k = Ctot.T @ k @ Ctot;


    nact = h.shape[0]

    h2 = pyscf.ao2mo.kernel(pymol, Ctot, aosym="s4", compact=False)
    h2.shape = (nact, nact, nact, nact)

    # The use of d1_embed only really makes sense if it has zero electrons in the
    # active space. Let's warn the user if that's not true

    S = pymol.intor("int1e_ovlp_sph")
    n_act = np.trace(S @ d1_embed @ S @ Ctot @ Ctot.T)
    if abs(n_act) > 1e-8 == False:
        print(n_act)
        error(" I found embedded electrons in the active space?!")

    h1 = h + j - .5*k;
    print(np.trace(Ctot @ Ctot.T @ F))
    return h0,h1,h2,S
h0,h1,h2,S=integral_making()

np.save("ints_h0", h0)
np.save("ints_h1", h1)
np.save("ints_h2", h2)
np.save("mo_coeffs", Ctot)
np.save("overlap_mat", S)
