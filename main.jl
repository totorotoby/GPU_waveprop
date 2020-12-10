using LinearAlgebra
using SparseArrays
using Kronecker
using Plots
using Printf
using CUDA
using CUDA.CUSPARSE
include("matfree_GPU.jl")



function mat_free_solve(c::Float64, x, y, t, Δx::Float64, Δy::Float64, Δt::Float64,
                        ni::Int, T::Int, U::Array{Float64,3}, F::Function)

    #@show ni
    #@show length(y)
    for (tind, tval) in enumerate(t[3:end])
        for (i, xval) in enumerate(x)
            for (j, yval) in enumerate(y)
                k = tind + 2
                #@show i,j,k
                # on south boundary
                if j == 1
                    d2y = (-2U[i, j , k-1] + U[i, j+1, k-1])/(Δy^2)
                # on north boundary
                elseif j == ni 
                    d2y = (U[i, j-1, k-1] - 2U[i,j,k-1])/(Δy^2)
                # interior
                else
                    d2y = (U[i, j-1, k-1] - 2U[i,j,k-1] + U[i, j+1, k-1])/(Δy^2)
                end
                # on east boundary
                if i == 1
                    d2x = (-2U[i,j,k-1] + U[i+1, j, k-1])/(Δx^2)
                # on west boundary
                elseif i == ni
                    d2x = (U[i-1, j, k-1] - 2U[i,j,k-1] )/(Δx^2)
                # interior
                else
                    d2x = (U[i-1, j, k-1] - 2U[i,j,k-1] + U[i+1, j, k-1])/(Δx^2)
                end

                # stencil computation
                U[i,j,k] = 2U[i,j,k-1] - U[i, j, k-2] + c^2 * (Δt^2) * (d2x + d2y) + (Δt^2)*F(xval, yval, tval)
            end
        end
    end
end



function knl_custom_mat_free!(c::Float64, x, y, t, Δx::Float64, Δy::Float64, Δt::Float64,
                            ni::Int, T, U, tind, samples)
                            #k is time index, passed as index of for loop calling this function

    bidx = blockIdx().x      #get thread's block ID
    tidx = threadIdx().x     #get thread ID
    dimx = blockDim().x      #get number of threads per block

    bidy = blockIdx().y      #thread's block ID
    tidy = threadIdx().y    #get thread ID
    dimy = blockDim().y     #get number of threads per block

    i = dimx * (bidx - 1) + tidx   #unique global thread ID in x-direction
    j = dimy * (bidy - 1) + tidy   #unique global thread ID in y-direction
    k = tind + 2

    if j <= ni && i <= ni
        #on south boundary
        if j == 1 && i <= ni
            d2y = (-2U[i, j, k-1] + U[i, j+1, k-1])/(Δy^2)
        #on north boundary
        elseif j == ni
            d2y = (U[i, j-1, k-1] - 2U[i,j,k-1])/(Δy^2)
        #interior
        else
            d2y = (U[i, j-1, k-1] - 2U[i,j,k-1] + U[i, j+1, k-1])/(Δy^2)
        end
        # on east boundary
        if i == 1
            d2x = (-2U[i,j,k-1] + U[i+1, j, k-1])/(Δx^2)
        # on west boundary
        elseif i == ni
            d2x = (U[i-1, j, k-1] - 2U[i,j,k-1] )/(Δx^2)
        # interior
        else
            d2x = (U[i-1, j, k-1] - 2U[i,j,k-1] + U[i+1, j, k-1])/(Δx^2)
        end

        # stencil computation
        sampleVal = samples[i, j]
        U[i,j,k] = 2U[i,j,k-1] - U[i, j, k-2] + c^2 * (Δt^2) * (d2x + d2y) + (Δt^2)*sampleVal 
    end

    return nothing
end


function knl_custom_mat_free_t!(c::Float64, M, Δx::Float64, Δy::Float64, Δt::Float64,
                            ni::Int, U, samples)
                            #k is time index, passed as index of for loop calling this function
    
    bidx = blockIdx().x      #get thread's block ID
    tidx = threadIdx().x     #get thread ID
    dimx = blockDim().x      #get number of threads per block

    bidy = blockIdx().y      #thread's block ID
    tidy = threadIdx().y    #get thread ID
    dimy = blockDim().y     #get number of threads per block

    i = dimx * (bidx - 1) + tidx   #unique global thread ID in x-direction
    j = dimy * (bidy - 1) + tidy   #unique global thread ID in y-direction
    
    
    if j <= ni && i <= ni
        for k in 3:M
            #on south boundary
            if j == 1 && i <= ni
                d2y = (-2U[i, j, k-1] + U[i, j+1, k-1])/(Δy^2)
                #on north boundary
            elseif j == ni
                d2y = (U[i, j-1, k-1] - 2U[i,j,k-1])/(Δy^2)
                #interior
            else
                d2y = (U[i, j-1, k-1] - 2U[i,j,k-1] + U[i, j+1, k-1])/(Δy^2)
            end
            # on east boundary
            if i == 1
                d2x = (-2U[i,j,k-1] + U[i+1, j, k-1])/(Δx^2)
                # on west boundary
            elseif i == ni
                d2x = (U[i-1, j, k-1] - 2U[i,j,k-1] )/(Δx^2)
                # interior
            else
                d2x = (U[i-1, j, k-1] - 2U[i,j,k-1] + U[i+1, j, k-1])/(Δx^2)
            end
            
            # stencil computation
            sampleVal = samples[i, j, k-1]
            U[i,j,k] = 2U[i,j,k-1] - U[i, j, k-2] + c^2 * (Δt^2) * (d2x + d2y) + (Δt^2)*sampleVal
            
            sync_threads()
        end
    end
        
    return nothing
end


function generateSamples(F, xin, yin, t, ni)    #generate samples of F on cpu for GPU kernel
    #sizeT = size(t)[1]
    sizeX = size(xin)[1]
    sizeY = size(yin)[1]
    sizeT = size(t)[1]
    samples = zeros(Float64, (sizeX, sizeY, sizeT))

    for (k, tval) in enumerate(t)
        for (i, xval) in enumerate(xin)
            for (j, yval) in enumerate(yin)
                samples[i, j, k] = F(xval, yval, tval)
            end
        end
    end
    
    #display(samples)
    return samples
end


function knl_gemv!(y, c, A, x, b)

    N = length(y)

    bid = blockIdx().x  # get the thread's block ID
    tid = threadIdx().x # get my thread ID
    dim = blockDim().x  # how many threads in each block

    i = dim * (bid - 1) + tid #unique global thread ID
        if i <= N
            for k = 1:N
                y[i] += c * A[i, k]*x[k]
            end
            y[i] += b[i]
        end
    return nothing
end



let

    m_refine = 1:4
    t_refine = 2:2:2
    nodes = 2 * 2 .^ m_refine
    #nodes = 96
    errors_CPU = Array{Float64,1}(undef, length(m_refine))
    errors_CPU_MOL = Array{Float64,1}(undef, length(m_refine))
    errors_GPU = Array{Float64,1}(undef, length(m_refine))
    errors_GPU_MOL = Array{Float64,1}(undef, length(m_refine))
    times_CPU = Array{Float64, 1}(undef, length(m_refine))
    times_CPU_MOL = Array{Float64, 1}(undef, length(m_refine))
    times_GPU = Array{Float64, 1}(undef, length(m_refine))
    times_GPU_MOL = Array{Float64, 1}(undef, length(m_refine))
    times_GPU_threads = Array{Float64, 1}(undef, length(t_refine))
    for (iter, nx) in enumerate(nodes)
        @printf("solving on %d x %d grid\n", nx, nx)
        # number of spatial nodes in x direction
        ny = nx
        # number of interior nodes
        ni = nx - 2
        #distance between nodes
        Δx = Δy = 2/(nx - 1)
        x = -1:Δx:1
        xin = x[2:nx-1]
        #@show collect(xin)
        y = -1:Δx:1
        yin = y[2:ny-1]
        # size of MOL vector solution
        N = ni^2

        # wave speed
        c = 1.0
        Δt = .5*Δx^2*c 
        T = 3
        t = 0:Δt:T
        M = length(t)
        @show M

        #################################
        #     Manufactured Solution     #
        #################################

        # guess solution sin waves scaled to domain
        ue(x,y,t) = sin(π*x)*sin(π*y)*cos(π*c*t)
        ue_xx(x,y,t) = -π^2*sin(π*x)*sin(π*y)*cos(π*c*t)
        ue_yy(x,y,t) = -π^2*sin(π*x)*sin(π*y)*cos(π*c*t)
        ue_t(x,y,t) = -π*c*sin(π*x)*sin(π*y)*sin(π*c*t)
        ue_tt(x,y,t) = -π^2*c^2*sin(π*x)*sin(π*y)*cos(π*c*t)
        
        F(x,y,t) = ue_tt(x,y,t) - c^2*(ue_xx(x,y,t) + ue_yy(x,y,t))


        # just interior points of source and guessed solution
        # matrix format
        ue_m(t, mesh) = [ue(i, j, t) for i in mesh, j in mesh]
        # column stacked
        ue_v(t, mesh) = [ue(i, j, t) for j in mesh for i in mesh]
        
        ue_tv(t, mesh) = [ue_t(i, j, t) for j in yin for i in mesh]
        F_v(t, mesh) = vcat(zeros(N), [F(i,j,t) for j in yin for i in mesh])
        F_m(mesh, ts) = [F(i, j, k) for i in mesh, j in mesh, k in ts]
        
        ###########################
        #    CPU Matrix-Free      # 
        ###########################
        #=

        @printf("Running CPU Matrix-Free verison\n")
        cpu_mf = @elapsed begin
            U = Array{Float64,3}(undef, ni, ni, M)
            U[:, :, 1] = ue_m(0, xin)
            U[:, :, 2] = ue_m(Δt, xin)
            
            mat_free_solve(c, xin, yin, t, Δx, Δy, Δt, ni, T, U, F)
        end
        times_CPU[iter] = cpu_mf
        errors_CPU[iter] = norm(U[:,:,end] .- ue_m(T, xin)) * sqrt(Δx^2)

        if iter != 1
            @printf("current error: %f\n", errors_CPU[iter])
            @printf("previous error: %f\n", errors_CPU[iter-1])
            @printf("rate: %f\n", log(2, errors_CPU[iter-1]/errors_CPU[iter]))
            
        end
        @printf("...done\n\n")
            
          =#  

            ###################
            #     CPU-MOL     #
            ###################
           #= 
            @printf("Running CPU-MOL verison\n")
        cpu_m = @elapsed begin
            Umol = Array{Float64,2}(undef, 2*N, M)
            # add displacement and velocity intial condition.
            Umol[:,1] = vcat(ue_v(0, xin), ue_tv(0, xin))
            #display(Umol[:,1])
            #discrete laplician
            Ix = sparse(I, ni, ni)
            Iy = sparse(I, ni, ni)
            D2 = sparse(1:ni, 1:ni, -2) +
                sparse(2:ni, 1:ni-1, ones(ni-1), ni, ni) +
                sparse(1:ni-1, 2:ni, ones(ni-1), ni, ni)
            Dxx = kron(D2, Iy)
            Dyy = kron(Ix, D2)
            
            # constructing matrix to do timestepping
            Auu = spzeros(N,N)
            Auv = sparse(I, N, N)
            Avu = (c^2/Δx^2) * (Dxx + Dyy)
            Avv = spzeros(N,N)
            A = [Auu Auv
                 Avu Avv]
            
            for m = 2:M
                Umol[:,m] = Umol[:,m-1] .+ Δt*c^2*(A*Umol[:,m-1]) #+ F_v((m-1)*Δt, xin))
            end
            
            usol = Umol[1:N,end]
        end
       
        
        plot_step = 4
        anim = @animate for m in 1:plot_step:M
            plot(xin, yin, Umol[1:N,m], st=:surface, zlims = (-1.5, 1.5))
            gui()
        end
        gif(anim, "waves.gif", fps = 15)
        
        times_CPU_MOL[iter] = cpu_m
        errors_CPU_MOL[iter] = norm(usol - ue_v(T, xin)) * √(Δx^2)
        
        if iter != 1
            @printf("current error: %f\n", errors_CPU_MOL[iter])
            @printf("previous error: %f\n", errors_CPU_MOL[iter-1])
            @printf("rate: %f\n", log(2, errors_CPU_MOL[iter-1]/errors_CPU_MOL[iter]))
        end

        @printf("...done\n\n")
        =#

        ##################################
        #   GPU Matrix-Free with Time    #
        ##################################
        if nx^2 <= 1024
            @printf("Running GPU Matrix-Free (Time inside) verison\n")
            
            h_U = zeros(ni, ni, M)
            h_U[:, :, 1] = ue_m(0, xin)
            h_U[:, :, 2] = ue_m(Δt, xin)
            
            d_U = CuArray(h_U)
            
            #ni * ni: total number of interior nodes
            
            num_threads_per_block_x = nx
            num_threads_per_block_y = nx
            num_blocks_x = cld(ni, num_threads_per_block_x)
            num_blocks_y = cld(ni, num_threads_per_block_y)
            
            
            tid_tuple = (num_threads_per_block_x, num_threads_per_block_y)
            bid_tuple = (num_blocks_x, num_blocks_y)
            
            #for (tind, tval) in enumerate(t[3:end]) #pass tind to kernel for K
            samples = F_m(xin, t)
            d_samples = CuArray(samples)
            @cuda threads=tid_tuple blocks=bid_tuple knl_custom_mat_free_t!(c, M, Δx, Δy, Δt,
                                                                          ni, d_U, d_samples)
            synchronize()

            Uout = Array(d_U)
            
            #=
            for k in 1:M
                plot(xin,yin, Uout[:,:,k],st=:surface, zlims = (-1.5, 1.5))
                gui()
            end
            =#
            
            
            #display(Uout[:,20:30,15:25])
            errors_GPU[iter] = norm(Uout[:,:,end] - ue_m(T, xin)) * √(Δx^2)

            
            if iter != 1
                @printf("current error: %f\n", errors_GPU[iter])
                @printf("previous error: %f\n", errors_GPU[iter-1])
                @printf("rate: %f\n", log(2, errors_GPU[iter-1]/errors_GPU[iter]))
            end
            @printf("...done\n\n")

        else
            @printf("Cannot run GPU-Mat Free with time because can only launch one block with physical nodes less than 1024")
        end

        

            
        #########################
        #   GPU Matrix-Free     #
        #########################
        
        # @printf("Running GPU Matrix-Free verison\n")
    
        # h_U = Array{Float64,3}(undef, ni, ni, M)
        # h_U[:, :, 1] = ue_m(0, xin)
        # h_U[:, :, 2] = ue_m(Δt, xin)


        # d_U = CuArray(h_U)
        # d_t = CuArray(t)
        # d_xin = CuArray(xin)
        # d_yin = CuArray(yin)
        # #ni * ni: total number of interior nodes

        # num_threads_per_block_x = 32
        # num_threads_per_block_y = 32
        # num_blocks_x = cld(ni, num_threads_per_block_x)
        # num_blocks_y = cld(ni, num_threads_per_block_y)

        
        # tid_tuple = (num_threads_per_block_x, num_threads_per_block_y)
        # bid_tuple = (num_blocks_x, num_blocks_y)

        # for (tind, tval) in enumerate(t[3:end]) #pass tind to kernel for K
        #     samples = generateSamples(F, xin, yin, tval, ni)
        #     d_samples = CuArray(samples)
        #     @cuda threads=tid_tuple blocks=bid_tuple knl_custom_mat_free!(c, d_xin, d_yin, d_t, Δx, Δy,
        #                                                                   Δt, ni, T, d_U, tind, d_samples)
        #     synchronize()
        # end
        
        # Uout = Array(d_U)
        # errors_GPU[iter] = norm(Uout[:,:,end] - ue_m(T, xin)) * √(Δx^2)

        
        # if iter != 1
        #     @printf("current error: %f\n", errors_GPU[iter])
        #     @printf("previous error: %f\n", errors_GPU[iter-1])
        #     @printf("rate: %f\n\n", log(2, errors_GPU[iter-1]/errors_GPU[iter]))
        #  end
        # @printf("...done\n")
        
        #=
        ###################
        #     GPU-MOL     #
        ###################
        function kel_F_v!(f_half, t, yin, xin)
            # with at lease (nx-2)*(ny-2) threads
            nyin = length(yin)
            nxin = length(xin)

            bid = blockIdx().x
            tid = threadIdx().x
            dim = blockDim().x

            ind = dim * (bid - 1) + tid

            if ind <= (nx - 2) * (ny - 2)
                indx = ind % nxin
                indy = trunc(ind / nyin) + 1
                x_t = xin[indx]
                y_t = yin[indy]

                f_half[ind] = π^2*c^2*sin(π*x_t)*sin(π*y_t)*cos(π*c*t)
                
            end
            return nothing
        end
        
        d_yin = CuArray(yin)
        d_xin = CuArray(xin)
        d_A = CuSparseMatrixCSR(A)
        num_threads_per_block = 32
        num_blocks = cld((nx-2)*(ny-2), num_threads_per_block)
        GPU_MOL_TIME = @elapsed begin
            @printf("Running GPU-MOL verison with threads per block = %d\n", num_threads_per_block)
            for m = 2:M
                f_half = zeros((nx-2)*(ny-2))
                d_f_half = CuArray(f_half)
                @cuda threads=num_threads_per_block blocks=num_blocks kel_F_v!(d_f_half, (m-1)*Δt, d_yin, d_xin)
                b = vcat(zeros(N), Array(d_f_half)) + Umol[:,m-1]
                d_b = CuArray(f_v)
                d_v = CuArray(Umol[:,m-1])
                # Umol[:,m] = Umol[:,m-1] .+ Δt*c^2*(A*Umol[:,m-1] + F_v((m-1)*Δt, xin))
                y = zeros(2 * (nx-2) * (ny-2))
                d_y = CuArray(y)
                @cuda threads=num_threads_per_block blocks=num_blocks knl_gemv!(d_y, Δt*c^2, d_A, d_v, d_b)
                Umol[:,m] = Array(y)
            end
        end
        
        times_GPU_MOL[iter] = GPU_MOL_TIME
        errors_GPU_MOL[iter] = norm(Array(d_y) - ue_v(T, xin)) * √(Δx^2)
        
        if iter != 1
            @printf("current error: %f\n", errors_CPU_MOL[iter])
            @printf("previous error: %f\n", errors_CPU_MOL[iter-1])
            @printf("rate: %f\n", log(2, errors_CPU_MOL[iter-1]/errors_CPU_MOL[iter]))
        end
        =#
        
        @printf("...done\n")

    end

    #plot(title = "Time by 

    
    #=
    plot(title = "speeds of different methods with mesh refining", xlabel = "log2 # grid points", ylabel = "time(s)")
    plot!(2:7, times_CPU, label="cpu stencil")
    plot!(2:7, times_CPU_MOL, label="cpu mol")
    plot!(2:7, times_GPU, label="gpu stencil")
    gui()
    =#
    #=
    times_CPU = Array{Float64, 1}(undef, length(m_refine))
    times_CPU_MOL = Array{Float64, 1}(undef, length(m_refine))
    times_GPU = Array{Float64, 1}(undef, length(m_refine))
    times_GPU_MOL = Array{Float64, 1}(undef, length(m_refine))
    =#
    

    
end




























        #=
        # Matrix Laplician for zero dirchelete
        Ix = sparse(I, nx, nx)
        Iy = sparse(I, nx, nx)
        D2 = sparse(1:nx, 1:nx, -2) +
            sparse(2:nx, 1:nx-1, ones(nx-1), nx, nx) +
            sparse(1:nx-1, 2:nx, ones(nx-1), nx, nx)
        Dxx = kron(D2, Iy)
        Dyy = kron(Ix, D2)
        Avu = (c^2/Δx^2) * (Dxx + Dyy)
        for i in 1:(nx*nx)

            if i <= nx
                Avu[i, :] = zeros(nx*nx)'
            end

            if i % nx == 0 || i % nx == 1
                 Avu[i, :] = zeros(nx*nx)'
            end

        end
        # Matrix Laplician for zero dirchelete
        Ix = sparse(I, nx, nx)
        Iy = sparse(I, nx, nx)
        D2 = sparse(1:nx, 1:nx, -2) +
            sparse(2:nx, 1:nx-1, ones(nx-1), nx, nx) +
            sparse(1:nx-1, 2:nx, ones(nx-1), nx, nx)
        Dxx = kron(D2, Iy)
        Dyy = kron(Ix, D2)
        Avu = (c^2/Δx^2) * (Dxx + Dyy)
        for i in 1:(nx*nx)
            if i <= nx
                Avu[i, :] = zeros(nx*nx)'
            end
            if i % nx == 0 || i % nx == 1
                 Avu[i, :] = zeros(nx*nx)'
            end
        end
        
        Uinit = ue_m(0,x)
        d_U = CuArray(Uinit)
        d_lap = Array{Float64, 2}(undef, nx, nx)
        d_lap = CuArray(d_lap)
        d_check = Avu * ue_v(0,x)
        d_check = reshape(d_check, (nx, nx))
        
        tpb = 32
        bn = cld(nx, tpb)
        @show nx, bn
        @cuda blocks=bn threads=tpb lap_1d_shared_kern!(d_U, d_lap, c, Δx,
                                                          nx, Val(tpb), Val(bn))
        
        display(Array(d_lap))
        display(d_check)
        
        @assert Array(d_lap) ≈ d_check
        
        =#

        
        #=
        # matrix free
        U = Array{Float64, 3}(undef, nx, nx, 3)
        U[:,:, 1] = ue_m(0, x)
        U[:,:, 2] = ue_m(Δt, x)
        d_U = CuArray(U)
        tpb = 4
        bn = cld(nx, tpb)
        
        @cuda blocks=bn threads=tpb shared_1d_kern!(d_U, c, Δx, Δt, nx, 4, Val(tpb), Val(bn))

        # matrix multiply
        Uinit = ue_v(0,x)
        Ucheck = ue_v(3*Δt, x)

        display(Array(d_U[:,:,3]))
        display(reshape(Ucheck, (nx,nx)))
        
        @assert reshape(Array(d_U[:,:,3]), (nx*nx,1))[1:32] ≈ Ucheck[1:32]

        =#

            
        ##################################
        #   GPU spatial Matrix-Free 2D   # 
        ##################################

        
        #=
        Three arrays to store spatial points at:
        current timestep
        1 timestep ago
        2 timesteps ago
        =#
        #=
        U1 = ue_m(0, x)
        U2 = ue_m(Δt, x)
        U3 = Array{Float64,2}(undef, nx, nx)

        #threads per block
        tpb = 32
        # number of blocks
        nb = cld(nx,32)

        # square dimensions of blocks
        thread_dims = (tpb, tpb)
        block_dims = (nb,nb)
        
        @printf("launching kernel with %f by %f blocks and %f by %f threads per block",
                block_dims[1], block_dims[2], thread_dims[1], thread_dims[2])

        for m = 3:4

            d_U = CuArray(U2)
            d_Uout = CuArray{Float64},(undef, nx, nx)
            @cuda blocks=block_dims threads=thread_dims !(d_U, c, Δx,
                                                                 Val(tpb), Val(nb))
            synchronize()
            #euler step
            U3[:,:] .= 2U2[:,:] .- U1[:,:] + Δt^2 .* Array(d_U)
            #shift everything over for next timestep
            U2[:,:] .= U3[:,:]
            U1[:,:] .= U2[:,:]
        end

        errors[iter] = norm(U[:,:,2] .- ue_m(T)) * sqrt(Δx^2)
        @printf("\tError mat free: %f\n", error)
        if iter != 1
            @printf("current error: %f\n", errors[iter])
            @printf("previous error: %f\n", errors[iter-1])
            @printf("rate: %f\n\n", log(2, errors[iter-1]/errors[iter]))
        end
        =#


        ################################################
        #   GPU spatial shared memory Matrix-Free 1D   # 
        ################################################

        
        #=
        Three arrays to store spatial points at:
        current timestep
        1 timestep ago
        2 timesteps ago
        =#
#=
        # Matrix Laplician for zero dirchelete
        Ix = sparse(I, nx, nx)
        Iy = sparse(I, nx, nx)
        D2 = sparse(1:nx, 1:nx, -2) +
            sparse(2:nx, 1:nx-1, ones(nx-1), nx, nx) +
            sparse(1:nx-1, 2:nx, ones(nx-1), nx, nx)
        Dxx = kron(D2, Iy)
        Dyy = kron(Ix, D2)
        Avu = (c^2/Δx^2) * (Dxx + Dyy)
        for i in 1:(nx*nx)

            if i <= nx
                Avu[i, :] = zeros(nx*nx)'
            end

            if i % nx == 0 || i % nx == 1
                 Avu[i, :] = zeros(nx*nx)'
            end

        end
        

        # matrix free
        U = ue_m(0, x)
        d_U = CuArray(U)
        Uout = Array{Float64, 2}(undef, nx, nx)
        d_Lap = CuArray(Uout)
        tpb = 32
        bn = cld(nx, tpb)
        
        #@show tpb, bn

        @cuda blocks=bn threads=tpb lap_1d_shared_kern!(d_U, d_Lap, c, Δx, nx, Val(tpb), Val(bn))

        # matrix multiply
        Uinit = ue_v(0,x)
        display(Uinit)
        #@show size(Uinit), size(Avu)
        Ucheck = Avu * Uinit
        
        
        @assert reshape(Uout, (nx*nx,1)) ≈ Ucheck
        
        
        #U2 = ue_m(Δt, x)
        #U3 = Array{Float64,2}(undef, nx, nx)

        #threads per block
        #tpb = 32
        # number of blocks
        #nb = cld(nx,32)

        # square dimensions of blocks
        #thread_dims = (tpb, tpb)
        #block_dims = (nb,nb)

        #=
        Ix = sparse(I, ni, ni)
        Iy = sparse(I, ni, ni)
        D2 = sparse(1:ni, 1:ni, -2) +
            sparse(2:ni, 1:ni-1, ones(ni-1), ni, ni) +
            sparse(1:ni-1, 2:ni, ones(ni-1), ni, ni)
        Dxx = kron(D2, Iy)
        Dyy = kron(Ix, D2)
        Avu = (c^2/Δx^2) * (Dxx + Dyy)
        =#
       

end
=#
    
