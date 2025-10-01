using LinearAlgebra
mutable struct Cluster
    orb_list::Vector{Int64}
    oei::Matrix{Float64}
    tei::Array{Float64,4}
    n_orb::Int
    p_evec::Union{Matrix{Float64}, Nothing}
    p_eval
    n_range::Int
    n_a::Int
    n_b::Int
    S2::Float64
    Ham::Float64
    tdm::Dict{Any, Any}
    heff::Float64
end

function Cluster(bl::Vector{Int64}, n_elec::Tuple{Int,Int}, oei::Matrix{Float64}, tei::Array{Float64,4}; Sz=nothing, S2=nothing)
    return Cluster(
        bl,
        oei,
        tei,
        size(oei, 1),
        nothing,
        nothing,
        0,
        n_elec[1],
        n_elec[2],
        S2 === nothing ? 0.0 : S2,
        0.0,
        Dict{Any, Any}(),
        0.0
    )
end



function tucker_vecs!(cl::Cluster, p::Matrix{Float64}, e)
    cl.p_evec = p
    cl.p_eval = e
end

function store_tdm!(cl::Cluster, key, tdm)
    cl.tdm[key] = tdm
end

function update_t!(cl::Cluster, heff::Float64)
    cl.h_eff += heff
end

mutable struct TuckerBlock
    veclist::Vector{Any}
    vector::Dict
    start::Int
    stop::Int
    shape::Int
    core::Int
end

function TuckerBlock(cl::Cluster)
    return TuckerBlock(Vector{Any}(undef, cl.n_orb), Dict{Int,Any}(), 0, 0, 0, 0)
end

function readvectors!(tb::TuckerBlock, vec, i)
    tb.vector[i] = vec
end

function vec_startstop!(tb::TuckerBlock, start, stop, shape)
    tb.start = start
    tb.stop = stop
    tb.shape = shape
end

function makecore!(tb::TuckerBlock, core)
    tb.core = core
end
