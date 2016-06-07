module slab
using MPI

export *

# Spectral transformation of three dimensional data aligned
# such that the last component is parallelized across processes

for (name, plan, fft12) in zip((:r2cSlab, :c2cSlab),
                               (:(FFTW.rFFTWPlan{T}), :(FFTW.cFFTWPlan)),
                               (:plan_rfft, :plan_fft))
@eval begin
    immutable ($name){T<:Real}
        # Global shape
        N::Array{Int, 1}
        # Global size of domain
        L::Array{T, 1}
        # Plans
        plan12::($plan)
        plan3::FFTW.cFFTWPlan
        # Work arrays for transformations
        vT::Array{Complex{T}, 3}
        vT_view::Array{Complex{T}, 4}
        v::Array{Complex{T}, 3}
        v_view::Array{Complex{T}, 4}
        v_recv::Array{Complex{T}, 3}
        v_recv_view::Array{Complex{T}, 4}
        # Communicator
        comm::MPI.Comm
        # Amount of data to be send by MPI
        chunk::Int    
        num_processes::Int

        # Constructor
        function ($name)(N, L, comm)
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
            plan12 = ($fft12)(A, (1, 2))
            plan3 = plan_fft!(v, (3, ))
            # Compute the inverse plans
            inv(plan12); inv(plan3)

            chunk = Nh*Np[2]*Np[3]
            # Now we are ready
            new(N, L, plan12, plan3,
                vT, vT_view, v, v_view, v_recv, v_recv_view,
                comm, chunk, p)
        end
    end

    # Constructor
    ($name){T<:Real}(N::Array{Int, 1}, L::Array{T, 1}, comm::Any) = ($name){T}(N, L, comm)

    # Transform real to complex as complex = T o real
    function rfft3{T<:Real}(F::($name){T}, fu::AbstractArray{Complex{T}, 3}, u::AbstractArray{T})
        A_mul_B!(F.vT, F.plan12, u)
        permutedims!(F.v_view, F.vT_view, [1, 2, 4, 3])
        MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
        F.plan3*F.v_recv; fu[:] = F.v_recv
    end

    # Transform complex to real as real = T o complex
    function irfft3{T<:Real}(F::($name){T}, u::AbstractArray{T}, fu::AbstractArray{Complex{T}, 3})
        F.plan3.pinv*fu; F.v[:] = fu
        MPI.Alltoall!(F.v_recv_view, F.v_view, F.chunk, F.comm)
        permutedims!(F.vT_view, F.v_recv_view, [1, 2, 4, 3])
        A_mul_B!(u, F.plan12.pinv, F.vT)
    end

    function real_shape{T<:Real}(F::($name){T})
        return (F.N[1], F.N[2], F.N[3]÷F.num_processes)
    end

    function complex_shape{T<:Real}(F::($name){T})
        return (F.N[1]÷2+1, F.N[2]÷F.num_processes, F.N[3])
    end

    end # begin
end # for
end # module
