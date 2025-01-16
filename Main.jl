# -*- coding: utf-8 -*
using StatsBase, Statistics, Distributions, Plots, Profile, ProfileSVG, FFTW
using ProgressMeter, EllipsisNotation, DataStructures
import Random
include("lib/tree.jl") # file with the probabilites tree dictionary
include("lib/coulomb.jl"); # file with the finding coulombian potential with the fourier transform 
include("lib/state.jl"); # file with the finding coulombian potential with the fourier transform 
# +
const a0 = 1  # distance between crystals (base unit)
const T = 1 # temperature (base unit)
const Lx = 10  # size of the sample
const Ly = 20
const Lz = 20
const eV = 38.681732675246586 # 1 eV in temperature units at 300 K

# Energy levels{
const Eg = 0.6*eV # bandgap
const E_std = 0.00*eV  # inhomogeneous broadening
const Ee_mean = 0.0 # ensemble-averaged electron energy in QD (upward from the metal Fermi level)
const Eh_mean = -Eg # ensemble-averaged hole energy in QD (downward from the metal Fermi level)
const EFl =-Eg + 0.0*eV #-Eg + 0.1*eV # Fermi energy in the left contact 
const EFr = -0.0*eV # Fermi energy in the right contact 
const dEbr = -0.1*eV # Artificial energy barrier reduction to reduce rejection rate with the same total injection rate

# Electron hopping rate (base unit)
const W0_min = 0.999  # electron energy spread boundaries
const W0_max = 1
const W_exc_rec = 0.0
# Columnb interation:
const e = 1 # elementary charge (base unit)
const Ec = 0.02*eV # coulomb interaction energy
const e_rad = 0.5*a0 # effective electron radius
const mlt = 4 # mesh multiplier, should be even
const coulomb = Coulomb(Lx, Ly, Lz, e_rad, mlt)

# Doping
base_doping = 0.1
doping = zeros(Lx, Ly, Lz)
doping[1:Lx÷2-1,:,:] .= base_doping;
doping[Lx÷2:end,:,:] .= base_doping;

const q = (none=0, e=-1, h=1, exc=0) # check!
const hops = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)] # possible directions of the lectron hops
const nhops = length(hops)
const injs = [(Lx, :e), (1, :e), (Lx, :h), (1, :h)];

# +
function move_r(r0::CartesianIndex{3}, dr::Tuple{Int64, Int64, Int64})  #function of a electron hop, it produces  Cartesian coordinates from the coordinate vectors 
    CartesianIndex(1 .+ mod.(Tuple(r0) .+ Tuple(dr) .- 1, (Lx, Ly, Lz))) #  when an electron passes through the boundaries of a sample, it appears at the opposite periodic boundary 
end
#as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...)); #

function mirror_r(r::CartesianIndex{3})
      return  CartesianIndex(2*Lx + 1 - r[1], r[2], r[3])
end;


# +
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
    
end
# -



# +
function fill_mini_tree!(W::AbstractArray)
    W[7,..] = W[1,..] + W[2,..]
    W[8,..] = W[3,..] + W[4,..]
    W[9,..] = W[5,..] + W[6,..]
    W[10,..] = W[7,..] + W[8,..] + W[9,..]
end

function sample_hop(w::AbstractVector, dv::Float64)
    if dv < w[7]
        if dv < w[1]
            return 1
        else
            return 2
        end
    else
        dv -= w[7]
        if dv < w[8]
            if dv < w[3]
                return 3
            else
                return 4
            end
        else
            dv -= w[8]
            if dv < w[5]
                return 5
            else
                return 6
            end   
        end
    end
end

# +
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

# +
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

# +
function potential(state::State, r::CartesianIndex{3}, exclude_id::Int=-1)
    P = state.static_phi[r] - (state.V + EFl - EFr)*(r[1] - 0.5)/Lx/e  # energy in r0 due to static potential and field
    for (k, (rc, qc)) in enumerate(zip(state.r, state.q))
        if k != exclude_id && qc!=0
            P +=  qc*coulomb[rc - r] # due to interaction between hopping charges
            rIc = mirror_r(rc) #cartesian coordinate of the image of charge in rc
            P += -qc*coulomb[rIc - r] # due to interaction with images of hopping charges
        end
    end 
    return P
end

function hope_rate(state::State, charge_id::Int, hop_num::Int) # Coulomb only
    dr = hops[hop_num]
    r0 = state.r[charge_id]
    q = state.q[charge_id] # charge
    if 0 < r0[1] + dr[1] <= Lx # usual hopping
        r1 = move_r(r0, dr)  # new cartesian coordinate of the electron
        dE = q*(state.static_phi[r1] - state.static_phi[r0] - dr[1]*(state.V + EFl - EFr)/Lx/e)  # energy between r0 and r1 due to static potential
        if q < 0 # electron
            dE += state.Ee[r1] - state.Ee[r0]
        else # hole
            dE -= state.Eh[r1] - state.Eh[r0]
        end
            
        for (k, (rc, qc)) in enumerate(zip(state.r, state.q))
            if k != charge_id && qc!=0
                dE += q*qc*(coulomb[rc - r1] - coulomb[rc - r0]) # energy difference between r0 and r1 due to interaction between hopping charges
                rIc = mirror_r(rc) # cartesian coordinate of the image of charge in rc
                dE += -q*qc*(coulomb[rIc - r1] - coulomb[rIc - r0]) # energy difference between r0 and r1 due to interaction with images of hopping charges
            end
        end 
        rI0 = mirror_r(r0)
        rI1 = mirror_r(r1)
        dE += -q^2*(coulomb[rI1 - r1] - coulomb[rI0 - r0]) # interaction with the image of hopped charge
    
    else # extraction
        E = q*potential(state, CartesianIndex(r0), charge_id)  # energy in r0 due to static potential and field               
        rI0 = mirror_r(r0)
        E += -q^2*coulomb[rI0 - r0]
        
        if q < 0 # electron
            E += state.Ee[r0]
        else # hole
            E -= state.Eh[r0]
        end
        
        if r0[1] == Lx
            dE = (EFl + state.V)*(-q/e) - E # actual potential on the right contact is EFl + state.V, it is not a mistake!
        else
            dE = EFl*(-q/e) - E
        end   
    end     
    
    return exp(-max(dE, 0)/T)
end

function injection_rate(state::State, r0::CartesianIndex{3}, t_in::Symbol)
    q_in = q[t_in]
    E = q_in*potential(state, r0)  # energy in r0 due to static potential and field               
    rI0 = mirror_r(r0)
    E += -q_in^2*coulomb[rI0 - r0]
    
    if t_in == :e # electron
        E += state.Ee[r0]
    else # hole
        E -= state.Eh[r0]
    end
    
    if r0[1] == Lx
        dE = E - (EFl + state.V)*(-q_in/e) # actual potential on the right contact is EFl + state.V, it is not a mistake!
    else
        dE = E - EFl*(-q_in/e)
    end
    
    return exp(-max(dE, 0)/T)*exp(dEbr/T) # dEbr makes the artifical decrease of the energy barrier to reduce the rejection rate
    
end;

# +
mutable struct Info   # collected information about the system
    frames::Int64
    time::Vector{Float64}
    cur_time::Float64
    trajectories::Array{Float64, 3}
    fillings::Array{Float64, 4}
    current::Vector{Float64}
    Info(state, frames) = initialize(new(), state, frames)
    total::DefaultDict{Symbol, Int64}    
end

function initialize(info::Info, state::State, frames::Int)
    info.frames = frames
    info.time = zeros(frames)
    info.cur_time = 0.0
    info.trajectories = zeros(frames, 3, state.N)
    info.fillings = zeros(frames, Lx, Ly, Lz)
    info.current = zeros(frames)
    info.total = DefaultDict{Symbol, Int64}(0)
    return info
end;
# -

function simulate(state::State, steps::Int, save_every::Int)  # main function, simulate current, calculate mean current
    
    frames = steps÷save_every

    info = Info(state, frames)
    
   @showprogress for frame in 1:frames
        sum_fillings = zeros(Lx, Ly, Lz)
        sum_dx = 0.0
        sum_dt = 0.0
        for substep in 1:save_every         
            dt, key, (ind, dv) = sample(state.trees)
            
            probe = rand()
            if key == :hop # e/h hop or exc recombination
                id = ind[1]
                r = state.r[id] # coordinate of the particle
                t = state.types[id] # type of the particle
                if (t==:e)||(t==:h)
                    info.total[:hop_trials] += 1
                    if t==:e
                      hop_rates = @view state.We_full[:, r]
                      hop_num = sample_hop(hop_rates , dv)
                    else
                        hop_num = sample_hop(state.Wh_full[:, r], dv)
                    end  
                    dr = hops[hop_num]              
                    #  @assert n[r0] # check that electron on r0 exists
                    if rand() < hope_rate(state, id, hop_num)
                        if 0 < (r[1] + dr[1]) && (r[1] + dr[1]) <= Lx # if el doesnt jumps throught the border
                            info.total[:hops] += 1
                            r1 = move_r(r, dr)  
                            id1 = state.ids[r1]
                            if id1 == 0
                                move_particle(state, id, r1)
                                sum_dx += q[t]*dr[1]
                            else
                                t1 = state.types[id1]
                                if (t == :e && t1 == :h)||(t == :h && t1 == :e) # recombination
                                    delete_particle(state, id)
                                    change_type(state, id1, :exc)
                                end
                            end
                        else
                            info.total[:extr] += 1
                            delete_particle(state, id) # extraction
                        end
                    end
                elseif t==:exc #recombination
                    info.total[:rec] += 1
                    delete_particle(state,id)
                end

             elseif key == :inj #injection
                info.total[:inj_trials] += 1
                inj, y, z = Tuple(ind)
                x, t_in = injs[inj]
                r1 = CartesianIndex(x,y,z)
                id1 = state.ids[r1]
                if id1 != 0
                    info.total[:inj_trials_occupied] += 1
                end
                rate = injection_rate(state, r1, t_in)
                if id1 == 0 && rate > 1
                    info.total[:inj_wrong_dEbr] += 1
                end
                if id1 == 0 && rand() < rate
                    info.total[:inj] += 1
                    add_particle(state, t_in, r1)
                end
                
             elseif key == :gen #thermal generation
                info.total[:gen] += 1
                if state.types[ind[1]] == :none
                     add_particle(state, :exc, ind[1])
                end
                                  
             end
            sum_dt += dt # time going past
            for (ri, qi) in zip(state.r, state.q)
                if qi!=0
                    sum_fillings[ri] += qi*dt
                end
            end

        end
        info.cur_time += sum_dt
        info.time[frame] = info.cur_time
        #info.trajectories[frame,:,:] = as_ints(state.r) # second index - x, y, z, third - el number
        info.fillings[frame,:,:,:] = sum_fillings/sum_dt; # frame, x, y ,z
        info.current[frame] = sum_dx/sum_dt*a0/Lx/Ly 
    end
    #mean_current = sum_dx/sum_dt*a0/Lx/Ly
    if info.total[:inj_wrong_dEbr] > 0
        @warn("dEbr is too big. info.total[:inj_wrong_dEbr] = $(info.total[:inj_wrong_dEbr])")
    end
    return info
end;

mutable struct stime
    st::Float64
end
S=stime(0)
V_range = range(-1*eV, 1*eV, 10)
current_vs_V = zeros(length(V_range))
for i = 1:length(V_range)
    Random.seed!(0)
    save_every = 1000
    state = State();
    state.V = V_range[i]
    info = simulate(state, 10*save_every, save_every)
    current_vs_V[i] = mean(info.current[500:end])
    S.st = info.cur_time + S.st
end

p = plot(V_range/eV, current_vs_V)
#plot!(V_range/eV, coeff.*V_range)
plot!(xlabel="V", ylabel="I")
plot!(size = (800, 600))
#plot!(xtickfontsize=14/2, ytickfontsize=14/2, xlabelfontsize=10, ylabelfontsize=10, thickness_scaling=3, legend=false )
#plot!(field_range[1:end],((-1 .+ exp.(field_range[1:end])*1.2)/30 .- 0.))
#savefig(p,"UI_pn_1000_1000_L20_10_10.png")
#savefig(p,"UI_pn_500_1000_L20_10_10.pdf")

(V_range[end]/a0/Lx)*W0_max*base_doping*a0^2/T


current_vs_V

μ = e*W0_max*a0^2/T
coeff = e*base_doping/a0^3*μ*Ly*Lz/(a0*Lx);

@time begin
    Random.seed!(0)
    save_every = 100;
    state = State();
    state.V = 1.6*eV
    info = simulate(state, 1000*save_every, save_every);
end
mean(info.current[500:end])

ProfileSVG.set_default(width=1200, timeunit=:s)
@profview simulate(state, 1000*save_every, save_every)

phys_a0 = 6 # real QD size, nm
phys_W0 = 1 # real rate W0, 1/ns

phi = [potential(state, CartesianIndex(x, 10, 1)) for x in 1:Lx]
plot(phi)

q_array = (i -> i==0 ? 0 : state.q[i]).(state.ids);
mean(q_array)

mean(q_array, dims=(2,3))
plot(mean(q_array, dims=(2,3))[:,1,1])
hline!([base_doping, -base_doping])

plot(mean(state.static_q, dims=(2,3))[:,1,1])

mean(state.static_phi, dims=(2,3))[1,1,1], mean(state.static_phi, dims=(2,3))[end,1,1]

size(transpose(mean(state.static_q, dims=3)[:,:,1]))

heatmap(transpose(mean(state.static_q, dims=3)[:,:,1]), colormap=:berlin, clim=(x->(-1,1).*maximum(abs, x)))

heatmap(transpose(mean(q_array,dims=3)[:,:,1]), colormap=:berlin, clim=(x->(-1,1).*maximum(abs, x)))
filter(x -> x > 0, q_array)
qn = q_array[:,:,:] .< 0 
qp = q_array[:,:,:] .> 0 
heatmap(transpose(mean(qp,dims=3)[:,:,1]), colormap=:berlin, clim=(x->(-1,1).*maximum(abs, x)))

heatmap(transpose(mean(qn,dims=3)[:,:,1]), colormap=:berlin, clim=(x->(-1,1).*maximum(abs, x)))

h1 = heatmap(q_array[1,:,:], colormap=:berlin, clim=(-1,1), aspect_ratio=:equal)
h2 = heatmap(q_array[end,:,:], colormap=:berlin, clim=(-1,1), aspect_ratio=:equal)
plot(h1, h2, size=(620, 300))

h = heatmap(phys_a0*(1:Lx), info.time, mean(info.fillings[:,:,:,:], dims=(3,4))[:,:,1,1], colormap=:berlin, clim=(x->(-0.5,0.5).*maximum(abs, x)))
plot!(xlabel = "x coordinate, nm")
plot!(ylabel = "time, ns", right_margin=10Plots.mm)
#savefig(h, "fillings_$(base_doping).png")

phi = potential(coulomb, q_array, 1, 1);

# +
a = 6 # real lattice parameter in nanometers
#heatmap(phi[Lx,:,:], right_margin=10Plots.mm)
#plot(phi[Lx+2,mlt*Int(Lz/2),:])
plot(a*(0.5:Lx), (Ee_mean .- e*mean(state.static_phi+phi, dims=(2,3))[:,1,1] + (0.5:Lx)*(state.V + EFl - EFr)/Lx)/eV, lw=2)
plot!(a*(0.5:Lx), (Eh_mean .- e*mean(state.static_phi+phi, dims=(2,3))[:,1,1] + (0.5:Lx)*(state.V + EFl - EFr)/Lx)/eV, lw=2) 
plot!(a*[-10, 0], [EFl, EFl]./eV, color="black", lw=2)
plot!(a*[Lx, Lx+10], [EFl + state.V, EFl + state.V]./eV, color="black", lw=2)
plot!(legend=false, xlabel="x coordinate, nm", ylabel="Energy, eV")


























































































































































