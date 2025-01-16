# -*- coding: utf-8 -*
using StatsBase, Statistics, Distributions, Plots, Profile, ProfileSVG, FFTW
using ProgressMeter, EllipsisNotation, DataStructures
import Random
include("lib/tree.jl") # file with the probabilites tree dictionary
include("lib/coulomb.jl"); # file with the finding coulombian potential with the fourier transform 
include("lib/state.jl"); # file with the finding coulombian potential with the fourier transform 
const a0 = 1  # distance between crystals (base unit)
const T = 1 # temperature (base unit)
const Lx = 20  # size of the sample
const Ly = 20
const Lz = 20
const eV = 38.681732675246586 # 1 eV in temperature units at 300 K

# Energy levels{
const Eg = 1.0*eV # bandgap
const E_std = 0.00*eV  # inhomogeneous broadening
const Ee_mean = 0.0 # ensemble-averaged electron energy in QD (upward from the metal Fermi level)
const Eh_mean = -Eg # ensemble-averaged hole energy in QD (downward from the metal Fermi level)
const Dlt = 0.001*eV #0.1*eV
const EFl = -Eg + Dlt # Fermi energy in the left contact 
const EFr = -Dlt # Fermi energy in the right contact 
const dEbr = 0.000*eV # 0.025*eV # Artificial energy barrier reduction to reduce rejection rate with the same total injection rate

# Electron hopping rate (base unit)
const W0_min = 0.999  # electron energy spread boundaries
const W0_max = 1
const W_exc_rec = 0.1
# Columnb interation:
const e = 1 # elementary charge (base unit)
const Ec = 0.02*eV # coulomb interaction energy
const e_rad = 0.5*a0 # effective electron radius
const mlt = 4 # mesh multiplier, should be even
const coulomb = Coulomb(Lx, Ly, Lz, e_rad, mlt)

# Doping
base_doping = 0.02
doping = zeros(Lx, Ly, Lz)
doping[1:Lx÷2-1,:,:] .= base_doping;
doping[Lx÷2:end,:,:] .= -base_doping;

const q = (none=0, e=-1, h=1, exc=0) # check!
const hops = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)] # possible directions of the lectron hops
const nhops = length(hops)
const injs = [(Lx, :e), (1, :e), (Lx, :h), (1, :h)]

function move_r(r0::CartesianIndex{3}, dr::Tuple{Int64, Int64, Int64})  #function of a electron hop, it produces  Cartesian coordinates from the coordinate vectors 
    CartesianIndex(1 .+ mod.(Tuple(r0) .+ Tuple(dr) .- 1, (Lx, Ly, Lz))) #  when an electron passes through the boundaries of a sample, it appears at the opposite periodic boundary 
end
#as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...)); #

function mirror_r(r::CartesianIndex{3})
      return  CartesianIndex(2*Lx + 1 - r[1], r[2], r[3])
end;

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
        info.fillings[frame,:,:,:] = sum_fillings/sum_dt; # frame, x, y ,z
        info.current[frame] = sum_dx/sum_dt*a0/Lx/Ly 
    end
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

p = plot(V_range/eV, current_vs_V);
plot!(xlabel="V", ylabel="I")
#plot!(size = (800, 600))
savefig(p,"UI_pn_1000_1000_L20_10_10.png")

#μ = e*W0_max*a0^2/T;
#coeff = e*base_doping/a0^3*μ*Ly*Lz/(a0*Lx);
phys_a0 = 6; # real QD size, nm
phys_W0 = 1; # real rate W0, 1/ns

@time begin
    Random.seed!(0)
    save_every = 100;
    state = State();
    state.V = 1.6*eV
    info = simulate(state, 1000*save_every, save_every)
end

#ProfileSVG.set_default(width=1200, timeunit=:s)
#@profview simulate(state, 1000*save_every, save_every)
h = heatmap(phys_a0*(1:Lx), info.time, mean(info.fillings[:,:,:,:], dims=(3,4))[:,:,1,1], colormap=:berlin, clim=(x->(-0.5,0.5).*maximum(abs, x)));
plot!(xlabel = "x coordinate, nm");
plot!(ylabel = "time, ns", right_margin=10Plots.mm);
savefig(h, "fillings_$(base_doping).png");

q_array = (i -> i==0 ? 0 : state.q[i]).(state.ids)
phi = potential(coulomb, q_array, 1, 1)
a = 6 # real lattice parameter in nanometers
f = plot(a*(0.5:Lx), (Ee_mean .- e*mean(state.static_phi+phi, dims=(2,3))[:,1,1] + (0.5:Lx)*(state.V + EFl - EFr)/Lx)/eV, lw=2);
plot!(a*(0.5:Lx), (Eh_mean .- e*mean(state.static_phi+phi, dims=(2,3))[:,1,1] + (0.5:Lx)*(state.V + EFl - EFr)/Lx)/eV, lw=2) 
plot!(a*[-10, 0], [EFl, EFl]./eV, color="black", lw=2)
plot!(a*[Lx, Lx+10], [EFl + state.V, EFl + state.V]./eV, color="black", lw=2)
plot!(legend=false, xlabel="x coordinate, nm", ylabel="Energy, eV")
savefig(f, "E(x).png");

























































































































































