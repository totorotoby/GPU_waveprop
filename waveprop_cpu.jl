using LinearAlgebra
using SparseArrays
using Kronecker
using Plots
using Printf
using CUDA
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
                    

let

    m_refine = 6:6
    nodes = 2*2 .^ m_refine
    errors = Array{Float64,2}(undef, length(m_refine), length(m_refine))

    
    for (iter, nx) in enumerate(nodes)
        @printf("%d number of elements\n", nx)
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
        T = 1
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
        ue_m(t, mesh) = [ue(i, j, t) for i in mesh, j in mesh]
        # column stacked
        ue_v(t, mesh) = [ue(i, j, t) for j in mesh for i in mesh]

        
        
        ue_tv(t, mesh) = [ue_t(i, j, t) for j in yin for i in mesh]
        F_v(t, mesh) = vcat(zeros(N), [F(i,j,t) for j in yin for i in mesh])
        
        #@show ue_m(0)
        #=
        for time in t
        z = [ue(i,j, time) for i in x, j in y]
        plot(x,y,z, st=:surface, zlims = (-1.5, 1.5))
        #Plots.contourf(x, y, z)
        sleep(.1)
        gui()
        end
        =#
        
        ###########################
        #    CPU Matrix-Free      # 
        ###########################
        #=
        U = Array{Float64,3}(undef, ni, ni, M)
        U[:, :, 1] = ue_m(0, xin)
        U[:, :, 2] = ue_m(Δt, xin)
        
        mat_free_solve(c, xin, yin, t, Δx, Δy, Δt, ni, T, U, F)
        #=
        for (m, time) in enumerate(t)
        plot(xin, yin, ue_m(time), st=:surface, c=:blues, zlims = (-(1+amp), 1+amp))
        plot!(xin, yin, U[:,:,m], st=:surface, zlims = (-(1+amp), 1+amp))
        sleep(.1)
        gui()
        end
        =#
        errors[iter] = norm(U[:,:,end] .- ue_m(T)) * sqrt(Δx^2)
        #@printf("\tError mat free: %f\n", error)
        if iter != 1
            @printf("current error: %f\n", errors[iter])
            @printf("previous error: %f\n", errors[iter-1])
            @printf("rate: %f\n\n", log(2, errors[iter-1]/errors[iter]))
            
        end

        end
        =#
        ###################
        #     CPU-MOL     #
        ###################
       #= 
        
        Umol = Array{Float64,2}(undef, 2*N, M)
        # add displacement and velocity intial condition.
        Umol[:,1] = vcat(ue_v(0), ue_tv(0))
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
            Umol[:,m] = Umol[:,m-1] .+ Δt*c^2*(A*Umol[:,m-1] + F_v((m-1)*Δt))
        end
        
        usol = Umol[1:N,end]

        #=
        plot_step = 4
        for m in 1:plot_step:M
        plot(xin, yin, Umol[1:N,m], st=:surface, zlims = (-(1+amp), 1+amp))
        sleep(.1)
        gui()
        end
        =#

        errors[iter] = norm(usol - ue_v(T)) * √(Δx^2)
        
        if iter != 1
            @printf("current error: %f\n", errors[iter])
            @printf("previous error: %f\n", errors[iter-1])
            @printf("rate: %f\n\n", log(2, errors[iter-1]/errors[iter]))
        end
        
        nothing
        
    end
    =#

            
        ##################################
        #     GPU spatial Matrix-Free    # 
        ##################################
        
        U = Array{Float64,3}(undef, nx, nx, 3)
        U[:, :, 1] = ue_m(0, x)
        U[:, :, 2] = ue_m(Δt, x)


        d_U
        display(U[:,:,2])

        tpb = 32
        nb = cld(nx,32)

        thread_dims = (tpd, tpb)
        block_dims = (nb,nb)

        
        @show thread_per_block, num_blocks
        GPU_lap(U, c, Δx)
        
        #=
        for (m, time) in enumerate(t)
        plot(xin, yin, ue_m(time), st=:surface, c=:blues, zlims = (-(1+amp), 1+amp))
        plot!(xin, yin, U[:,:,m], st=:surface, zlims = (-(1+amp), 1+amp))
        sleep(.1)
        gui()
        end
        =#
        #errors[iter] = norm(U[:,:,end] .- ue_m(T)) * sqrt(Δx^2)
        #@printf("\tError mat free: %f\n", error)
        #if iter != 1
        #    @printf("current error: %f\n", errors[iter])
        #    @printf("previous error: %f\n", errors[iter-1])
        #    @printf("rate: %f\n\n", log(2, errors[iter-1]/errors[iter]))
        #    
        #end

    end

end
    
