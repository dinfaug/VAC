mutable struct State   # describe system state
    dt::Float64 # time step
    r::Vector{CartesianIndex{3}} # coordinates of the particles (all)
    types::Vector{Symbol}
    q::Vector{Int}
    ids::Array{Int, 3}
    We_full::Array{Float64, 4}
    Wh_full::Array{Float64, 4}
    W::Vector{Float64} # tunneling rates for the particles
    Winj::Array{Float64, 3}
    Wrec::Array{Float64, 3}
    Wgen::Array{Float64, 3}
    
    N::Int64 # number of the particles
    static_q::Array{Int64, 3} # charge of the particles
    static_phi::Array{Float64, 3} # static potential
    static_phi_fine::Array{Float64, 3}
    Ee::Array{Float64, 3} # electron energies in QDs
    Eh::Array{Float64, 3} # hole energies in QDs
    V::Float64 # applied voltage (energy units)
    trees::Dict{Symbol, RateTree} # tree of rates 
    free_ids::Vector{Int}
    State() = initialize(new()) # structure with the current information about the system 
end

function initialize(state::State)
    mask = rand(Lx, Ly, Lz) .< abs.(doping)  # random space distribution of the particles (e or h)
    state.r = findall(mask) # array of particle coordinates
    state.N = length(state.r)  # number of particle
    state.types = fill(:none, state.N)
    for (id, r) in enumerate(state.r)
        if doping[r] > 0
            state.types[id] = :e
        else
            state.types[id] = :h
        end
    end
    state.q = [q[t] for t in state.types]
    state.ids = zeros(Int, Lx, Ly, Lz)
    state.ids[mask] = 1:state.N
    state.free_ids = []
    
    state.We_full = zeros(10, Lx, Ly, Lz) # rates depending on initial and final coordinat  
    state.Wh_full = zeros(10, Lx, Ly, Lz)
    
    state.W = zeros(state.N)
    state.Winj = zeros(4, Ly, Lz)
       
    state.Ee = Ee_mean .+ Ec*randn(Lx, Ly, Lz)
    state.Eh = Eh_mean .+ Ec*randn(Lx, Ly, Lz)
    
    state.Wrec = fill(W_exc_rec, Lx, Ly, Lz)
    state.Wgen =  @.  state.Wrec*exp(-(state.Ee - state.Eh)/T)
      
    doping_mlt = repeat(doping/mlt^3, inner=(mlt,mlt,mlt))
    static_mask = rand(mlt*Lx, mlt*Ly, mlt*Lz) .< abs.(doping_mlt)  # random space distribution of the charges
    state.static_q = zeros(mlt*Lx, mlt*Ly, mlt*Lz) # matrix of charges
    state.static_q[static_mask] .= sign.(doping_mlt[static_mask])    
    
    state.static_phi_fine = potential(coulomb, state.static_q, mlt, mlt)
    state.static_phi = potential(coulomb, state.static_q, mlt, 1)
    
    
#### Filling of the dictionary W with the probabilities of the hops between all centers           

    for r in CartesianIndices(zeros(Lx,Ly,Lz))
        for i in 1:2:nhops
            rnd = rand(LogUniform(W0_min, W0_max))
            state.We_full[i,r] === rnd
           #print(move_r(r,hops[i]))
            state.We_full[i+1, move_r(r,hops[i])] === state.We_full[i,r]
            state.Wh_full[i,r] === rnd
            state.Wh_full[i+1, move_r(r,hops[i])] === state.Wh_full[i,r]
        end
    end
    # Make mini-tree
    fill_mini_tree!(state.We_full)    
    fill_mini_tree!(state.Wh_full)  
    
    for id in 1:state.N
        state.W[id] = get_hop_rate(state, id)
    end
    for y in 1:Ly
        for z in 1:Lz
            for inj in 1:4
                state.Winj[inj, y, z] = get_inj_rate(state, inj, y, z)
            end
        end
    end

    state.trees = Dict()
    state.trees[:hop] = RateTree(state.W)
    state.trees[:inj] = RateTree(state.Winj) 
    state.trees[:gen] = RateTree(state.Wgen)    
   
    return state
    
end;

# injection 

function get_inj_rate(state::State, inj::Int, y::Int, z::Int)
    x, t = injs[inj]
    if x == 1
        if t==:e
            rate = state.We_full[2, 1, y, z]
        else
            rate = state.Wh_full[2, 1, y, z]
        end
    else
        if t==:e
            rate = state.We_full[1, Lx, y, z]
        else
            rate = state.Wh_full[1, Lx, y, z]
        end
    end
    rate *= exp(-dEbr/T)
    
    id1 = state.ids[x, y, z]
    if id1 == 0
        return rate
    else
        t1 = state.types[id1]
        if (t==:e && t1==:h)||(t==:h && t1==:e)
            return rate # exciton formation
        else
            return 0.0
        end
    end
end

function update_inj_rate(state::State, r::CartesianIndex)
    x, y, z = Tuple(r)
    for (inj, (x1, t1)) in enumerate(injs)
        if x == x1
            state.trees[:inj][inj, y, z] = get_inj_rate(state, inj, y, z)
        end
    end
end

# free ids

function add_free_ids(state::State, n::Int)
    append!(state.r, zeros(CartesianIndex{3}, n))
    append!(state.types, fill(:none, n))
    append!(state.q, zeros(Int, n))
    append!(state.free_ids, state.N .+ (1:n))
    
    W = zeros(state.N + n)
    W[1:state.N] = get_rates(state.trees[:hop])
    
    state.trees[:hop] = RateTree(W)
    
    state.N += n
end

function get_free_id(state::State)
    if length(state.free_ids) == 0
        add_free_ids(state, 100)
    end
    return pop!(state.free_ids)
end

function get_hop_rate(state::State, id::Int)
    t = state.types[id]
    if t==:e
        return state.We_full[10, state.r[id]]
    elseif t==:h
        return state.Wh_full[10, state.r[id]]
    elseif t==:exc
        return state.Wrec[state.r[id]]
    else
        return 0.0
    end    
end

# particles

function update_hop_rate(state::State, id::Int)
    state.trees[:hop][id] = get_hop_rate(state, id)

end

function change_type(state::State, id::Int, new_t::Symbol)
    state.types[id] = new_t
    state.q[id] = q[new_t]
    update_hop_rate(state, id)
    update_inj_rate(state, state.r[id])
end

function move_particle(state::State, id::Int, new_r::CartesianIndex)
    state.ids[state.r[id]] = 0
    state.ids[new_r] = id
    update_inj_rate(state, state.r[id])
    update_inj_rate(state, new_r)
    state.r[id] = new_r
    update_hop_rate(state, id)
end

function delete_particle(state::State, id::Int)
    change_type(state, id, :none)
    state.ids[state.r[id]] = 0 # ??
    push!(state.free_ids, id)
end

function add_particle(state::State, new_t::Symbol, new_r::CartesianIndex)
    id = get_free_id(state)
    state.ids[new_r] = id
    state.r[id] = new_r
    change_type(state, id, new_t)
end