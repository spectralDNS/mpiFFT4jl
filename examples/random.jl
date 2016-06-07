using MPI
MPI.Init()

import mpiFFT4jl

function foo(comm::MPI.Comm)
    const N = [32, 64, 64]    # Global shape of mesh
    const L = [2pi, 2pi, 2pi] # Real size of mesh
    
    FFT = mpiFFT4jl.slab.SlabFFT(N, L, comm)

    # Create a random array and its transform
    U = rand(Float64, mpiFFT4jl.slab.real_shape(FFT))
    U_hat = Array{Complex{Float64}}(mpiFFT4jl.slab.complex_shape(FFT))
    U2 = similar(U)

    for i in 1:60
        mpiFFT4jl.slab.rfft3(FFT, U_hat, U)
        mpiFFT4jl.slab.irfft3(FFT, U2, U_hat)
    end

    MPI.Reduce(sumabs2(U2-U), MPI.SUM, 0, comm) 
    
end

comm = MPI.COMM_WORLD
@time k = foo(comm)
if MPI.Comm_rank(comm) == 0 println("Error = $(k)") end
MPI.Finalize()
