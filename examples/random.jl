using MPI
MPI.Init()

import mpiFFT4jl

function foo(comm::MPI.Comm)
    const N = [32, 64, 64]    # Global shape of mesh
    const L = [2pi, 2pi, 2pi] # Real size of mesh
    
    FFT = mpiFFT4jl.slab.r2c(N, L, comm)

    # Create a random array and its transform
    U = rand(Float64, mpiFFT4jl.slab.real_shape(FFT))
    U_hat  = Array{Complex{Float64}}(mpiFFT4jl.slab.complex_shape(FFT))
    U_hat2 = similar(U_hat)
    U2 = similar(U)
    U3 = similar(U)
    
    mpiFFT4jl.slab.rfft3(FFT, U_hat, U)

    for i in 1:10
        mpiFFT4jl.slab.irfft3(FFT, U2, U_hat)
#         mpiFFT4jl.slab.irfft3(FFT, U3, U_hat, 3)
    end

    (MPI.Reduce(sumabs2(U2-U), MPI.SUM, 0, comm), FFT)
end

comm = MPI.COMM_WORLD
@time (k, FFT) = foo(comm)
if MPI.Comm_rank(comm) == 0 println("Error  = $(k)") end
MPI.Finalize()
