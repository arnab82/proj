using Einsum

function get_hubbard_params(n_site, beta, U, pbc=true)
    """
    Generates the parameters for a linear Hubbard model.

    Args:
        n_site (int): Number of sites in the system.
        beta (float): Hubbard hopping parameter.
        U (float): Hubbard on-site interaction parameter.
        pbc (bool): Whether to apply periodic boundary conditions.

    Returns:
        h_local (ndarray): Local part of the Hubbard Hamiltonian.
        g_local (ndarray): Interaction part of the Hubbard Hamiltonian.
    """
    """It initializes an empty matrix t of shape (n_site, n_site) to represent the hopping matrix.
    It fills the hopping matrix t with 1s to represent the nearest-neighbor hopping between sites.
    If periodic boundary conditions (pbc) are enabled, it adds the hopping term between the first and last sites.
    It computes the local part of the Hubbard Hamiltonian (h_local) by multiplying t with the negative hopping parameter beta.
    It initializes an empty tensor g_local of shape (n_site, n_site, n_site, n_site) to represent the interaction matrix.
    It fills the diagonal elements of g_local with the on-site interaction parameter U.
    It returns the local and interaction parts of the Hubbard Hamiltonian (h_local and g_local, respectively).
    The get_hubbard_params function allows you to generate the necessary parameters for a linear Hubbard model based on the number of sites, the hopping parameter, and the on-site interaction parameter."""
    # gets the interactions for linear hubbard
    println(" ---------------------------------------------------------")
    println("                       Hubbard model")
    println(" ---------------------------------------------------------")
    println(" nsite: ", n_site)
    println(" beta : ", beta)
    println(" U    : ", U)

    t = zeros(n_site, n_site)

    for i in 1:n_site-1
        t[i, i+1] = 1
        t[i+1, i] = 1
    end
    if pbc
        t[n_site, 1] = 1
        t[1, n_site] = 1
    end

    h_local = -beta .* t

    g_local = zeros(n_site, n_site, n_site, n_site)
    for i in 1:n_site
        g_local[i, i, i, i] = U
    end

    return h_local, g_local
end
function run_hubbard_scf(h_local, g_local, closed_shell_nel, do_scf=true)
    """
    Performs the delocalized mean-field calculation for a Hubbard model.

    Args:
        h_local (ndarray): Local part of the Hubbard Hamiltonian.
        g_local (ndarray): Interaction part of the Hubbard Hamiltonian.
        closed_shell_nel (int): Number of electrons in the closed-shell part of the system.
        do_scf (bool): Whether to perform self-consistent field calculation.

    Returns:
        Escf (float): Mean-field energy.
        orb (ndarray): Orbital energies.
        h (ndarray): Transformed local part of the Hubbard Hamiltonian.
        g (ndarray): Transformed interaction part of the Hubbard Hamiltonian.
        C (ndarray): Transformation matrix.
    """
    """It performs the self-consistent field (SCF) calculation if do_scf is True. Otherwise, it assumes the input h_local as the diagonalized form of the Hamiltonian and sets the transformation matrix C as the identity matrix.
    It computes the transformed local part of the Hubbard Hamiltonian h by applying the transformation matrix C to h_local.
    It applies the transformation matrix C to the interaction part of the Hubbard Hamiltonian g_local using the einsum function.
    It performs a series of contractions using einsum to transform the indices of the interaction tensor g based on the transformation matrix C.
    It defines slices o and v to represent the occupied and virtual orbital indices, respectively.
    It computes the mean-field energy Escf by evaluating the energy expression using the transformed Hamiltonian elements.
    It returns the mean-field energy Escf, orbital energies orb, transformed local Hamiltonian h, transformed interaction Hamiltonian g, and the transformation matrix C.
    The run_hubbard_scf function allows you to perform the mean-field calculation for a Hubbard model, obtaining the mean-field energy and the transformed Hamiltonian matrices."""
    println()
    println(" ---------------------------------------------------------")
    println("                  Delocalized Mean-Field")
    println(" ---------------------------------------------------------")
    if do_scf
        orb, C = eigen(h_local)
    else
        C = Matrix{Float64}(I, size(h_local, 1), size(h_local, 1))
        orb = diag(h_local)
    end

    println("Orbital energies:")
    println(orb)
    println()

    h = C' * h_local * C

    @einsum g[l,q,r,s]=g_local[p,q,r,s]*C[p,l] 
    @einsum g[l,m,r,s]=g[l,q,r,s]*C[q,m]
    @einsum g[l,m,n,s]=g[l,m,r,s]*C[r,n] 
    @einsum g[l,m,n,o]=g[l,m,n,s]*C[s,o] 

    o = 1:closed_shell_nel
    v = closed_shell_nel+1:size(h_local, 1)

    Escf = 2 * sum(h[o, o]) + 2 * sum(g[o, o, o, o]) - sum(g[o, o, o, o])

    println("Mean Field Energy        : ", Escf)

    println(C)

    return Escf, orb, h, g, C
end
function get_hubbard_1d(n_site, beta1, beta2, U, pbc=true)
    """The function takes the following input parameters:

    n_site: Number of lattice sites in the 1D system.
    beta1: Hopping parameter for even sites.
    beta2: Hopping parameter for odd sites.
    U: On-site interaction strength.
    pbc (optional): Boolean flag indicating whether to apply periodic boundary conditions (default is true).
    The function first performs some assertions and prints information about the Hubbard model, such as the number of lattice sites (n_site), hopping parameters (beta1 and beta2), and the on-site interaction strength (U).
    It initializes an empty hopping matrix t of size n_site × n_site.
    Using a loop, it assigns the hopping parameters to the appropriate positions in the t matrix based on the parity of the lattice site indices. Even sites receive beta1, while odd sites receive beta2. If pbc is true, the last and first sites are also connected with beta2.
    It creates a deep copy of the hopping matrix t and assigns it to h_local. This ensures that the original hopping matrix remains unmodified.
    It initializes an empty interaction tensor g_local of size n_site × n_site × n_site × n_site.
    Using another loop, it sets the on-site interaction strength U in g_local for each lattice site.
    Finally, the function returns the hopping matrix h_local and the interaction tensor g_local.
    The purpose of this function is to generate the local part (h_local) and interaction part (g_local) of the Hubbard Hamiltonian for a 1D system with the specified parameters."""
    println(n_site ÷ 2)
    @assert n_site % 2 == 0

    # Gets the interactions for linear Hubbard
    println(" ---------------------------------------------------------")
    println("                       Hubbard model")
    println(" ---------------------------------------------------------")
    println(" nsite : ", n_site)
    println(" beta1 : ", beta1)
    println(" beta2 : ", beta2)
    println(" U     : ", U)

    t = zeros(n_site, n_site)

    for i in 1:n_site-1
        if i % 2 == 0
            t[i, i+1] = -beta1
            t[i+1, i] = -beta1
        else
            t[i, i+1] = -beta2
            t[i+1, i] = -beta2
        end
    end

    if pbc
        t[n_site, 1] = -beta2
        t[1, n_site] = -beta2
    end

    h_local = copy(t)
    g_local = zeros(n_site, n_site, n_site, n_site)
    for i in 1:n_site
        g_local[i, i, i, i] = U
    end

    return h_local, g_local
end
function make_2d_lattice(dim_a, dim_b, beta1, beta2, U)
    """The function takes the following input parameters:

    dim_a: The number of lattice sites along the a dimension.
    dim_b: The number of lattice sites along the b dimension.
    beta1: The hopping parameter for even sites.
    beta2: The hopping parameter for odd sites.
    U: The on-site interaction strength.
    It calculates the total number of lattice sites n_site by multiplying dim_a and dim_b.
    It initializes the hopping matrix t as a zero matrix of size n_site × n_site.
    The function uses nested loops to assign the hopping parameters beta1 and beta2 to the appropriate positions in t based on the parity of the lattice site indices:
    For each lattice site at coordinates (a, b), it calculates the corresponding index ind using (a-1) * dim_b + b.
    If a is even, it calculates the index ind2 for the neighboring site (a+1, b) and assigns beta1 to t[ind, ind2] and t[ind2, ind].
    If a is odd, it calculates the index ind2 for the neighboring site (a+1, b) and assigns beta2 to t[ind, ind2] and t[ind2, ind].
    It performs a similar calculation for the neighboring site (a, b+1) and assigns beta1 if b is even and beta2 if b is odd and not at the edge.
    It initializes the interaction tensor g_local as a zero tensor of size n_site × n_site × n_site × n_site.
    The function assigns the on-site interaction strength U to the diagonal elements of g_local by setting g_local[i, i, i, i] = U for each lattice site index i.
    Finally, the function returns the hopping matrix t and the interaction tensor g_local.
    This function is useful for constructing the Hamiltonian matrices for a two-dimensional lattice with hopping terms and on-site interactions."""
    n_site = dim_a * dim_b
    t = zeros(n_site, n_site)
    
    for a in 1:dim_a
        for b in 1:dim_b
            ind = (a-1) * dim_b + b
            if a % 2 == 0
                ind2 = a * dim_b + b
                t[ind, ind2] = beta1
                t[ind2, ind] = beta1
            else
                ind2 = a * dim_b + b
                try
                    t[ind, ind2] = -beta2
                    t[ind2, ind] = -beta2
                catch
                end
            end
            
            if b % 2 == 0
                ind2 = (a - 1) * dim_b + b + 1
                t[ind, ind2] = beta1
                t[ind2, ind] = beta1
            elseif b % 2 == 1 && b % dim_b != 0
                ind2 = (a - 1) * dim_b + b + 1
                t[ind, ind2] = -beta2
                t[ind2, ind] = -beta2
            end
        end
    end
    
    g_local = zeros(n_site, n_site, n_site, n_site)
    
    for i in 1:n_site
        g_local[i, i, i, i] = U
    end
    
    return t, g_local
end
function make_stack_lattice(dim_a, dim_b, beta1, beta2, U, pbc=true)
    """
    make a lattice with strong intra in 1d and weak in 2nd dimension
    for 4 x 3
    it makes a cube in strong interaction and stacks them on top of each other

    dima = 3
    dimb = 4 pbc in dim b
    this will be 12 site stacked cube
    """
    """The function takes several input parameters: dim_a and dim_b define the dimensions of the lattice, beta1 and beta2 specify the hopping parameters, U represents the on-site interaction strength, and pbc (optional) indicates whether periodic boundary conditions should be applied in the second dimension.
    The total number of lattice sites, n_site, is calculated as the product of dim_a and dim_b.
    The hopping matrix t is initialized as a square matrix of zeros with dimensions n_site × n_site.
    The function iterates over each lattice site using nested for loops. For each site with indices a and b, the corresponding indices in the t matrix are determined as ind = a * dim_b + b and ind2 = (a + 1) * dim_b + b.  
    The function assigns the hopping parameters based on the site indices and periodic boundary conditions. If b is less than dim_b - 1, the hopping parameter beta1 is assigned between the current site and the next site in the same row. If b is equal to dim_b - 1 and pbc is true, the hopping parameter beta1 is assigned between the current site and the first site in the next row. Additionally,
the hopping parameter beta2 is assigned between the current site and the corresponding site in the next column.
    The interaction tensor g_local is initialized as a tensor of zeros with dimensions n_site × n_site × n_site × n_site. The diagonal elements of g_local are set to the value of U to represent the on-site interaction.
    Finally, the function returns the hopping matrix t and the interaction tensor g_local as the output. 
    Overall, the make_stack_lattice function allows you to construct a specific lattice configuration with desired hopping parameters and interactions"""
    n_site = dim_a * dim_b
    t = zeros(Float64, (n_site, n_site))
    for a in 0:dim_a-1
        for b in 0:dim_b-1
            ind = a * dim_b + b + 1
            ind2 = (a + 1) * dim_b + b + 1
            if ind2 <= n_site
                t[ind, ind2] = beta2
                t[ind2, ind] = beta2
            end

            if b < dim_b - 1
                ind2 = a * dim_b + b + 2
                if ind2 <= n_site
                    t[ind, ind2] = beta1
                    t[ind2, ind] = beta1
                end
            elseif pbc && a > 0
                ind2 = (a - 1) * dim_b + b + 2
                if ind2 <= n_site
                    t[ind, ind2] = beta1
                    t[ind2, ind] = beta1
                end
            end
        end
    end

    g_local = zeros(Float64, (n_site, n_site, n_site, n_site))
    for i in 1:n_site
        g_local[i, i, i, i] = U
    end

    return t, g_local
end
