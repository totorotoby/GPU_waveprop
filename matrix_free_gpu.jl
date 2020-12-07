using LinearAlgebra
using SparseArrays
using Kronecker
using Plots
using CUDA


#from class code
function get_interior(y, N)
    return yint = y[2:N-1, 2:N-1][:]
end


function knl_custom_mat_free!(c, x, y, t, U, Δx, Δy, Δt, ni, T, F, tind)
    #k is time index, passed as index of for loop calling this function
    

    bid = blockIdx().x      #get thread's block ID
    tid = threadIdx().x     #get thread ID
    dim = blockDim().x      #get number of threads per block

    #bidy = blockIdx().y      #thread's block ID
    #tidy = threadIdx().y    #get thread ID
    #dimy = blockDim().y     #get number of threads per block

    i = dim * (bid - 1) + tid   #unique global thread ID in x-direction
    
    
    #j = dimy * (bidy - 1) + tidy   #unique global thread ID in y-direction
    k = tind + 2
    



    j = 1
    #for loop for j until x/y thread blocks implemented. fixing kernel error first
    for yval in y
        
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
        U[i,j,k] = 2U[i,j,k-1] - U[i, j, k-2] + c^2 * (Δt^2) * (d2x + d2y) #+ (Δt^2)*F(xval, yval, tval)
        j += 1
    end
    
    return nothing
    
end



function mat_free_solve(c::Float64, x, y, t, Δx::Float64, Δy::Float64, Δt::Float64,
                        ni::Int, T, U::Array{Float64,3}, F::Function)
                        #T::Int

    #@show ni
    #@show length(y)
    for (tind, tval) in enumerate(t[3:end])  #this will be the for loop calling the kernel function
        for (i, xval) in enumerate(x)   #this will be the threads
            for (j, yval) in enumerate(y)   #this will need to be a for loop in the kernel
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


let

    m_refine = 4:16
    errors = Array{Float64,2}(undef, length(m_refine), length(m_refine))
    

    for nx in 34:34#enumerate(m_refine)
        # number of spatial nodes in x direction
        ny = nx
        # number of interior nodes
        ni = nx - 2
        #distance between nodes
        Δx = Δy = 2/(nx - 1)

        x = -1:Δx:1
        xin = x[2:nx-1]
        y = -1:Δx:1
        yin = y[2:ny-1]
        
        # size of MOL vector solution
        N = ni^2
        
        # wave speed
        c = 1.0
        # time step
        Δt = .5*Δx^2*c
        T = .5
        t = 0:Δt:T
        M = length(t)

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
        ue_m(t) = [ue(i, j, t) for i in xin, j in yin]
        # column stacked
        ue_v(t) = [ue(i, j, t) for j in yin for i in xin]


        ue_tv(t) = [ue_t(i, j, t) for j in yin for i in xin]
        F_v(t) = vcat(zeros(N), [F(i,j,t) for j in yin for i in xin])

        #########################
        #   GPU Matrix-Free     #
        #########################

        h_U = Array{Float64,3}(undef, ni, ni, M)
        h_U[:, :, 1] = ue_m(0)
        h_U[:, :, 2] = ue_m(Δt)

        d_U = CuArray(h_U)
        d_t = CuArray(t)
        d_xin = CuArray(xin)
        d_yin = CuArray(yin)
        #println("finished cuarray allocations")
        #ni * ni: total number of interior nodes
        #totalNodes = (ni + 2)^2

        num_threads_per_block = 32
        #num_threads_per_block_y = 32
        num_blocks = cld(ni, num_threads_per_block)
        #num_blocks_y = cld(ni, num_threads_per_block_y)
        
        t_start = time()
        for (tind, tval) in enumerate(t[3:3]) #pass tind to kernel for K
            @cuda blocks=num_blocks threads=num_threads_per_block knl_custom_mat_free!(c, d_xin, d_yin, d_t, d_U, Δx, Δy, Δt, ni, T, F, tind)
            synchronize()
        end
        t_device = time() - t_start
        #println("gpu time: ",t_device)

        #d_error = norm(d_U[:,:,end] .- ue_m(T)) * sqrt(Δx^2)
        #println("Error gpu: ", d_error)

        #calculate speed up rate cpu/gpu here
        #speedUp = t_device / t_host

    end
end
