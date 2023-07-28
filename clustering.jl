using LinearAlgebra

mutable struct Cluster
    orb_list::Int
    oei::Matrix{Float64}
    tei::Matrix{Float64}
    n_orb::Int
    p_evec::Matrix{Float64}
    p_eval::Vector{Float64}
    n_range::Int
    n_a::Int
    n_b::Int
    S2::Float64
    Ham::Float64
    tdm::Dict{Any,Any}
    h_eff::Float64
end

function init!(cl::Cluster, bl::Int, n_elec::Tuple{Int,Int}, oei::Matrix{Float64}, tei::Matrix{Float64}, Sz::Union{Nothing,Float64}=nothing, S2::Union{Nothing,Float64}=nothing)
    cl.orb_list = bl
    cl.n_a = n_elec[1]
    cl.n_b = n_elec[2]
    cl.oei = oei
    cl.tei = tei
    cl.n_orb = size(oei, 1)
end

function tucker_vecs!(cl::Cluster, p::Matrix{Float64}, e::Vector{Float64})
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
