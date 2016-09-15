include("utils.jl")
using MPI
using .utils

# export rfft3, irfft3, real_shape, complex_shape, r2c
export *

# Spectral transformation of three dimensional data aligned
# such that the last component is parallelized across processes
type r2c{T<:Real}
    # Global shape
    N::Array{Int, 1}
    # Global size of domain
    L::Array{T, 1}
    # Communicator
    comm::MPI.Comm
    num_processes::Int
    rank::Int
    chunk::Int    # Amount of data to be send by MPI
    
    # Plans
    plan12::FFTW.rFFTWPlan{T}
    plan3::FFTW.cFFTWPlan
    # Work arrays for transformations
    vT::Array{Complex{T}, 3}
    vT_sub::Array{Complex{T}, 4}
    v::Array{Complex{T}, 3}
    v_sub::Array{Complex{T}, 4}
    v_recv::Array{Complex{T}, 3}
    v_recv_sub::Array{Complex{T}, 4}
    dealias::Array{Int, 1}

    # Constructor
    function r2c(N, L, comm)
        # Verify input
        Nh = N[1]÷2+1
        p = MPI.Comm_size(comm)
        Np = N÷p

        # Allocate work arrays
        vT, v = Array{Complex{T}}(Nh, N[2], Np[3]), Array{Complex{T}}(Nh, Np[2], N[3])
        vT_view, v_view = reshape(vT, (Nh, Np[2], p, Np[3])), reshape(v, (Nh, Np[2], Np[3], p))
        # For MPI.Alltoall! preallocate the receiving buffer
        v_recv = similar(v); v_recv_view = reshape(v_recv, (Nh, Np[2], Np[3], p))

        # Plan Fourier transformations
        A = zeros(T, (N[1], N[2], Np[3]))
        if p > 1
            plan12 = plan_rfft(A, (1, 2))
            plan3 = plan_fft!(v, (3, ))
        else  # Use only plan12 to do entire transform
            plan12 = plan_rfft(A, (1, 2, 3))
            plan3 = plan_fft!(zeros(Complex{Float64}, 2,2,2), (1, 2, 3))
        end

        # Compute the inverse plans
        inv(plan12)
        if p > 1 inv(plan3) end

        chunk = Nh*Np[2]*Np[3]
        # Now we are ready
        new(N, L, comm, p, MPI.Comm_rank(comm), chunk,
            plan12, plan3,
            vT, vT_view, v, v_view, v_recv, v_recv_view)
    end
end

# Constructor
r2c{T<:Real}(N::Array{Int, 1}, L::Array{T, 1}, comm::Any) = r2c{T}(N, L, comm)

# Transform real to complex as complex = T o real
function rfft3{T<:Real}(F::r2c{T}, fu::AbstractArray{Complex{T}, 3}, u::AbstractArray{T})
    if F.num_processes > 1
        A_mul_B!(F.vT, F.plan12, u)
        permutedims!(F.v_view, F.vT_view, [1, 2, 4, 3])
        MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
        F.plan3*F.v_recv; fu[:] = F.v_recv
    else
        A_mul_B!(fu, F.plan12, u)
    end
end

# Transform complex to real as real = T o complex
function irfft3{T<:Real}(F::r2c{T}, u::AbstractArray{T}, fu::AbstractArray{Complex{T}, 3}, dealias_fu::Int=0)
    if F.num_processes > 1
        F.v[:] = fu
        if dealias_fu == 1
            dealias(F, F.v)
        elseif dealias_fu == 2
            dealias2(F, F.v)
        end
        F.plan3.pinv*F.v
        MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
        permutedims!(F.vT_view, F.v_recv_view, [1, 2, 4, 3])
        A_mul_B!(u, F.plan12.pinv, F.vT)
    else
        A_mul_B!(u, F.plan12.pinv, fu)
    end
end

function real_shape{T<:Real}(F::r2c{T})
    (F.N[1], F.N[2], F.N[3]÷F.num_processes)
end

function complex_shape{T<:Real}(F::r2c{T})
    (F.N[1]÷2+1, F.N[2]÷F.num_processes, F.N[3])
end

function complex_shape_T{T<:Real}(F::r2c{T})
    (F.N[1]÷2+1, F.N[2], F.N[3]÷F.num_processes)
end

function complex_local_slice{T<:Real}(F::r2c{T})
    ((1, F.N[1]÷2+1), 
     (F.rank*F.N[2]÷F.num_processes+1, (F.rank+1)*F.N[2]÷F.num_processes),
     (1, F.N[3]))
end

function complex_local_wavenumbers{T<:Real}(F::r2c{T})
    (rfftfreq(F.N[1], 1.0/F.N[1]),
     fftfreq(F.N[2], 1.0/F.N[2])[F.rank*div(F.N[2], F.num_processes)+1:(F.rank+1)*div(F.N[2], F.num_processes)],
     fftfreq(F.N[3], 1.0/F.N[3]))
end

function get_local_wavenumbermesh{T<:Real}(F::r2c{T})
    K = Array{Int}(tuple(push!([complex_shape(F)...], 3)...))
    k = complex_local_wavenumbers(F)
    for (i, Ki) in enumerate(ndgrid(k[1], k[2], k[3])) K[sub(i)...] = Ki end
    K
end

function get_local_mesh{T<:Real}(F::r2c{T})
    # Real grid
    x = collect(0:F.N[1]-1)*F.L[1]/F.N[1]
    y = collect(0:F.N[2]-1)*F.L[2]/F.N[2]
    z = collect(0:F.N[3]-1)*F.L[3]/F.N[3]
    X = Array{T}(tuple(push!([real_shape(F)...], 3)...))
    for (i, Xi) in enumerate(ndgrid(x, y, z[F.rank*F.N[3]÷F.num_processes+1:(F.rank+1)*F.N[3]])) X[sub(i)...] = Xi end
    X
end    

function dealias{T<:Real}(F::r2c{T}, fu::AbstractArray{Complex{T}, 3})
    kk = complex_local_wavenumbers(F)
    for (k, kz) in enumerate(kk[3])
        x = false
        if abs(kz) > div(F.N[3], 3)
        @inbounds fu[:, :, k] = 0.0
            continue
        end
        for (j, ky) in enumerate(kk[2])
            if abs(ky) > div(F.N[2], 3)
               @inbounds fu[:, j, k] = 0
                continue
            end
            for (i, kx) in enumerate(kk[1])
                if (abs(kx) > div(F.N[1], 3))
                    @inbounds fu[i, j, k] = 0.0
                end
            end
        end
    end
end

function dealias2{T<:Real}(F::r2c{T}, fu::AbstractArray{Complex{T}, 3})
    if  !isdefined(F, :dealias)
        const kmax_dealias = F.N/3
        K = get_local_wavenumbermesh(F)
        (kx, ky, kz) = K[:,:,:,1], K[:,:,:,2], K[:,:,:,3]
        indices = []
        i = 1
        for (x,y,z) in zip(kx, ky, kz)
            if abs(x) > div(F.N[1], 3) || abs(y) > div(F.N[2], 3) || abs(z) > div(F.N[3], 3)
                push!(indices, i)
            end
            i += 1
        end
        F.dealias = indices
    end
    for i in F.dealias
      @inbounds  fu[i] = 0.0
    end
end


