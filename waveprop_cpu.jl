using LinearAlgebra
using SparseArrays
using Kronecker
using Plots



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
                U[i,j,k] = 2U[i,j,k-1] - U[i, j, k-2] + c * (Δt^2) * (d2x + d2y) + (Δt^2)*F(xval, yval, tval)
            end
        end
    end
end


let

    # number of spatial nodes in x direction
    nx = ny = 12
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
    ###################
    #  derive CFL??   #
    ###################
    Δt = .01
    T = 3
    t = 0:Δt:T
    M = length(t)
    
    # wave speed
    c = 1.0

    #################################
    #     Manufactured Solution     #
    #################################
 
    # gaussian width
    b_w = .5
    # gaussian height
    amp = 1

    # gaussian
    f(x) = amp * exp(-((x) / b_w)^2)
    fp(x) = -2((x) / b_w^2)* amp * exp(-((x) / b_w)^2)
    fpp(x) = -(2 * amp * exp(-(x)^2/b_w^2)*(b_w^2 - 2x^2))/b_w^4

    # convert to radial coordinates
    rad(x,y) = sqrt(x^2 + y^2)
    rad_x(x,y) = x / sqrt(x^2 + y^2)
    rad_xx(x,y) = y^2 / sqrt(x^2 + y^2)
    rad_y(x,y) = y / sqrt(x^2 + y^2)
    rad_y(x,y) = x^2 / sqrt(x^2 + y^2)

    #=
    guessed solution
    sending plane wave out a reflection back in same direction.
    could use radial waves instead with rad, but getting source function
    will be annoying...
    =#

    ue(x,y,t) = f(x+y - c * t) -
        f(x + y - 3 + c * t)
    ue_xx(x,y,t) = fpp(x+y - c*t) -
        fpp(x + y - 3 + c*t)
    ue_yy(x,y,t) = fpp(x+y - c*t) -
        fpp(x + y - 3 + c*t)
    ue_t(x,y,t) = -c*(fp(x+y - c*t)
                         +fp(x+y - 3 + c*t))
    ue_tt(x,y,t) = c^2*(fpp(x+y - c*t) -
                        fpp(x + y - 3 + c*t))
    
    F(x,y,t) = ue_tt(x,y,t) - c*(ue_xx(x,y,t) + ue_yy(x,y,t))


    # just interior points of source and guessed solution
    # matrix format
    ue_m(t) = [ue(i, j, t) for i in xin, j in yin]
    # column stacked
    ue_v(t) = [ue(i, j, t) for j in yin for i in xin]
    ue_tv(t) = [ue_t(i, j, t) for j in yin for i in xin]
    F_v(t) = vcat(zeros(N), [F(i,j,t) for j in yin for i in xin])
    
    #=
    for time in t
        z = [ue(i,j, time) for i in x, j in y]
        plot(x,y,z, st=:surface, zlims = (-(1+amp), 1+amp))
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
    U[:, :, 1] = ue_m(0)
    U[:, :, 2] = ue_m(Δt)
    mat_free_solve(c, xin, yin, t, Δx, Δy, Δt, ni, T, U, F)
    
    for (m, time) in enumerate(t)
        plot(xin, yin, U[:,:,m], st=:surface, zlims = (-(1+amp), 1+amp))
        sleep(.1)
        gui()
    end
    
    #error = norm(U[:, :, end] - ue_v)
    =#
    ###################
    #     CPU-MOL     #
    ###################

    Umol = Array{Float64,2}(undef, 2*N, M)

    # add displacement and velocity intial condition.
    Umol[:,1] = vcat(ue_v(0), ue_tv(0))
    
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
    Avu = (c/Δx^2)*(Dxx + Dyy)
    Avv = spzeros(N,N)
    
    A = [Auu Auv
         Avu Avv]

    for m = 2:M
        Umol[:,m] = Umol[:,m-1] .+ Δt*(A*Umol[:,m-1] + F_v((m-1)*Δt))
    end

    usol = Umol[1:N,end]

    error = norm(usol - ue_v(T)) * √(Δx^2)

    @show error
    
    
    nothing

    
    
end
