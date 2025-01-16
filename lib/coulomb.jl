using OffsetArrays



function Coulomb_kernel(Lx, Ly, Lz, e_rad)
        K = zeros(Lx, Ly, Lz)
        kw_max = 5 # maximum number of the periods in k-space
        Nx = Int(round(kw_max/e_rad*Lx/2/pi))
        Ny = Int(round(kw_max/e_rad*Ly/2/pi))
        Nz = Int(round(kw_max/e_rad*Lz/2/pi))
        for nx in -Nx:Nx
            for ny in -Ny:Ny
                for nz in -Nz:Nz
                    kx = 2*pi*nx/Lx
                    ky = 2*pi*ny/Ly
                    kz = 2*pi*nz/Lz
                    k2 = kx^2 + ky^2 + kz^2
                    K[mod(nx, Lx) + 1, mod(ny, Ly) + 1, mod(nz, Lz) + 1] += Ec*4*pi*exp(-k2*e_rad^2/2)/k2
                end
            end
        end
        K[1, 1, 1] = 0
        return K
    end

struct Coulomb   # makes new data type for tree
    kernel::Array{Float64, 3}
    kernel_fine::Array{Float64, 3}
    coulomb :: OffsetArray{Float64, 3, Array{Float64, 3}}
    function Coulomb(Lx, Ly, Lz, e_rad, mlt)
        
        kernel = Coulomb_kernel(2*Lx, Ly, Lz, e_rad)
        kernel_fine = Coulomb_kernel(2*mlt*Lx, mlt*Ly, mlt*Lz, mlt*e_rad)*mlt;
       
        C = real(ifft(kernel))
        #  Filling the array C with the potentials of the probe charge (q=1 at (0,0,0) center) at the all centers in the sample, its reflection
        #  Coulomb_kernel is the function from Coulomb_Fourier.jl, that calculates fourier transform of the distances between the centers
        coulomb = OffsetArray(repeat(C, outer=(2, 2, 2)), -2*Lx:2*Lx-1, -Ly:Ly-1, -Lz:Lz-1) 
        # Filling the array with the potentials of the probe charge (q=1 at (0,0,0) center)                                                                                     # at the all centers in the sample, its reflection and in n copies of both
        
        return new(kernel, kernel_fine, coulomb) 
    end 
end

function Base.getindex(c::Coulomb, r::CartesianIndex{3})
    return c.coulomb[r]
end


function potential(C::Coulomb, rho, mlt_rho, mlt_phi)
    if mlt_rho == 1
        rho_fine = zeros(2*Lx*mlt, Ly*mlt, Lz*mlt)
        rho_fine[mlt÷2:mlt:end, mlt÷2:mlt:end, mlt÷2:mlt:end] = cat(rho, -rho[end:-1:1,:,:], dims=1)
    else
        rho_fine = cat(rho[1:end-1,:,:], zeros(1, Ly*mlt, Lz*mlt), -rho[end-1:-1:1,:,:], zeros(1, Ly*mlt, Lz*mlt), dims=1)
    end
    phi_fine = real(ifft(C.kernel_fine.*fft(rho_fine)));
    phi_fine_half = phi_fine[1:Lx*mlt, :, :]
    if mlt_phi == 1
        return phi_fine_half[mlt÷2:mlt:end, mlt÷2:mlt:end, mlt÷2:mlt:end]
    else
        return phi_fine_half
    end
end
