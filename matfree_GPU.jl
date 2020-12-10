using CUDA



function shared_1d_kern!(U, c, Δx, Δt, n, M, ::Val{TILE_DIM}, ::Val{BLOCK_NUM}) where {TILE_DIM, BLOCK_NUM}

    tid = threadIdx().x
    bid = blockIdx().x

    i = (bid - 1) * TILE_DIM + tid

    # initialize shared memory
    if bid == 1 || bid == BLOCK_NUM
        grid = @cuStaticSharedMem(Float64, (TILE_DIM + 1, 3, 3))
    else
        grid = @cuStaticSharedMem(Float64, (TILE_DIM + 2, 3, 3))
    end
    
    for k in 3:M
        for j in 2:n-1
            if k == 3
                #############################
                #      load shared memory   #
                #############################
                # north boundary
                if bid == 1
                    # halo node
                    if tid == TILE_DIM
                        for wd in 1:3
                            g = wd - 2
                            for t in 1:2
                                # load my indices
                                grid[tid, wd, t] = U[i, j-g, t]
                                # halo outside
                                grid[tid + 1, wd, t] = U[i+1, j-g, t]
                            end
                        end
                    else
                        for wd in 1:3
                            g = wd - 2 
                            for t in 1:2
                                # load my indices
                                grid[tid, wd, t] = U[i,j-g, t]
                            end
                        end
                    end
                    
                # south boundary
                elseif bid == BLOCK_NUM
                    # halo node
                    if tid == 1
                        for wd in 1:3
                            g = wd - 2 
                            for t in 1:2
                                # load my indices
                                grid[tid + 1, wd, t] = U[i, j-g, t]
                                # halo outside
                                grid[tid, wd, t] = U[i-1,j-g, t]
                            end
                        end
                    else
                        for wd in 1:3
                            g = wd - 2
                            for t in 1:2
                                # load my indices
                                grid[tid + 1, wd, t] = U[i,j-g, t]
                            end
                        end
                    end
                else
                    if tid == 1
                        for wd in 1:3
                            g = wd - 2 
                            for t in 1:2
                                # load my indices
                                grid[tid + 1, wd, t] = U[i,j-g, t]
                                # halo outside
                                grid[tid, wd, t] = U[i-1,j-g, t]
                            end
                        end
                    elseif tid == TILE_DIM
                        for wd in 1:3
                            g = wd - 2 
                            for t in 1:2
                                # load my indices
                                grid[tid + 1, wd, t] = U[i,j-g, t]
                                # halo outside
                                grid[tid + 2, wd, t] = U[i+1,j-g, t]
                            end
                        end
                    else
                        for wd in 1:3
                            g = wd - 2 
                            for t in 1:2
                                # load my indices
                                grid[tid + 1, wd, t] = U[i,j-g, t]
                            end
                        end
                    end
                end
                sync_threads()
            end

            
            ######################################
            #    shift shared memory/dont load   #
            ######################################
            
            if k > 3
                # north boundary
                if bid == 1
                    # halo node
                    if tid == TILE_DIM
                        for wd in 1:3
                            for t in 2:3
                                # load my indices
                                grid[tid, wd, t-1] = grid[tid, wd, t]
                                # halo outside
                                grid[tid + 1, wd, t-1] = grid[tid + 1, wd, t] 
                            end
                        end
                    else
                        for wd in 1:3
                            for t in 2:3
                                # load my indices
                                grid[tid, wd, t-1] = grid[tid, wd, t]
                            end
                        end
                    end
                    # south boundary
                elseif bid == BLOCK_NUM
                    # halo node
                    if tid == 1
                        for wd in 1:3
                            for t in 2:3
                                # load my indices
                                grid[tid + 1, wd, t-1] =  grid[tid + 1, wd, t]
                                # halo outside
                                grid[tid, wd, t-1] =  grid[tid, wd, t]
                            end
                        end
                    else
                        for wd in 1:3
                            for t in 2:3
                                # load my indices
                                grid[tid + 1, wd, t-1] = grid[tid + 1, wd, t]
                            end
                        end
                    end
                else
                    if tid == 1
                        for wd in 1:3
                            for t in 2:3
                                # load my indices
                                grid[tid + 1, wd, t-1] = grid[tid + 1, wd, t]
                                # halo outside
                                grid[tid, wd, t-1] = grid[tid, wd, t]
                            end
                        end
                    elseif tid == TILE_DIM
                        for wd in 1:3
                            for t in 2:3
                                # load my indices
                                grid[tid + 1, wd, t-1] = grid[tid + 1, wd, t]
                                # halo outside
                                grid[tid + 2, wd, t-1] = grid[tid + 2, wd, t]
                            end
                        end
                    else
                        for wd in 1:3
                            for t in 2:3
                                # load my indices
                                grid[tid + 1, wd, t-1] = grid[tid + 1, wd, t]
                            end
                        end
                    end 
                end
               sync_threads() 
            end

            ##################################
            #   Compute with shared memory   # 
            ##################################
            # if on north boundary
            if bid == 1
                if tid == 1
                    #@cuprintln("looking at $i, $j")
                    U[i, j, 3] = grid[tid, 2, 2]
                else
                    U[i, j, 3] = 2*grid[tid, 2, 2] - grid[tid, 2, 1] +
                        (Δt^2*c^2/Δx^2) * (grid[tid + 1, 2, 2] + grid[tid - 1, 2, 2] +
                                           grid[tid, 3, 2] + grid[tid, 1, 2] - 4grid[tid, 2, 2])
                end

            # if on south boundary
            elseif bid == BLOCK_NUM
                if tid == TILE_DIM
                    U[i,j, 3] = grid[tid + 1, 2, 2]
                    
                else
                    U[i,j, 3] = 2grid[tid+1, 2, 2] - grid[tid+1,2,1] +
                        (Δt^2*c^2/Δx^2) * (grid[tid + 2, 2, 2] + grid[tid, 2, 2] +
                                           grid[tid + 1, 3, 2] + grid[tid + 1, 1, 2] - 4grid[tid + 1, 2, 2])
                end
            # if on interior
            else
                U[i, j, 3] = 2grid[tid+1, 2, 2] - grid[tid+1, 2, 1] +
                    (Δt^2*c^2/Δx^2) * (grid[tid + 2, 2, 2] + grid[tid, 2, 2] +
                                       grid[tid + 1, 3, 2] + grid[tid + 1, 1, 2] - 4grid[tid + 1, 2, 2])
            end
            sync_threads()
        end
    end
    return nothing
end


function lap_1d_shared_kern!(U, Lap, c, Δx, n, ::Val{TILE_DIM}, ::Val{BLOCK_NUM}) where {TILE_DIM, BLOCK_NUM}

    tid = threadIdx().x
    bid = blockIdx().x

    i = (bid - 1) * TILE_DIM + tid
   
    # initialize shared memory
    if bid == 1 || bid == BLOCK_NUM
        grid = @cuStaticSharedMem(Float64, (TILE_DIM + 1, 3))
    else
        grid = @cuStaticSharedMem(Float64, (TILE_DIM + 2, 3))
    end
    
    for j in 2:n-1

        #############################
        #      load shared memory   #
        #############################
        
        # north boundary
        if bid == 1
            # halo node
            if tid == TILE_DIM
                # load my indices
                grid[tid, 1] = U[i,j-1]
                grid[tid, 2] = U[i,j]
                grid[tid, 3] = U[i,j+1]
                # halo outside
                grid[tid + 1, 1] = U[i+1,j-1]
                grid[tid + 1, 2] = U[i+1,j]
                grid[tid + 1, 3] = U[i+1,j+1]
            else
                # load my indices
                grid[tid, 1] = U[i,j-1]
                grid[tid, 2] = U[i,j]
                grid[tid, 3] = U[i,j+1]

            end

        # south boundary
        elseif bid == BLOCK_NUM
            # halo node
            if tid == 1
                # load my indices
                grid[tid + 1, 1] = U[i,j-1]
                grid[tid + 1, 2] = U[i,j]
                grid[tid + 1, 3] = U[i,j+1]
                # halo outside
                grid[tid, 1] = U[i-1,j-1]
                grid[tid, 2] = U[i-1,j]
                grid[tid, 3] = U[i-1,j+1]
            else
                # load my indices
                grid[tid + 1, 1] = U[i,j-1]
                grid[tid + 1, 2] = U[i,j]
                grid[tid + 1, 3] = U[i,j+1]
            end
        else
            if tid == 1
                # load my indices
                grid[tid + 1, 1] = U[i,j-1]
                grid[tid + 1, 2] = U[i,j]
                grid[tid + 1, 3] = U[i,j+1]
                # halo outside
                grid[tid, 1] = U[i-1,j-1]
                grid[tid, 2] = U[i-1,j]
                grid[tid, 3] = U[i-1,j+1]

            elseif tid == TILE_DIM
                # load my indices
                grid[tid + 1, 1] = U[i,j-1]
                grid[tid + 1, 2] = U[i,j]
                grid[tid + 1, 3] = U[i,j+1]
                # halo outside
                grid[tid + 2, 1] = U[i+1,j-1]
                grid[tid + 2, 2] = U[i+1,j]
                grid[tid + 2, 3] = U[i+1,j+1]

            else
                # load my indices
                grid[tid + 1, 1] = U[i,j-1]
                grid[tid + 1, 2] = U[i,j]
                grid[tid + 1, 3] = U[i,j+1]
            end
            
        end
        
        sync_threads()
  
        
        ##################################
        #   Compute with shared memory   # 
        ##################################
            # if on north boundary
        if bid == 1
            if tid == 1
                Lap[i,j] = grid[tid, 2]
            else
                Lap[i,j] = (c^2/Δx) * grid[tid + 1, 2] + grid[tid - 1, 2] +
                    grid[tid, 3] + grid[tid, 1] - 4grid[tid, 2]
            end
            # if on south boundary
        elseif bid == BLOCK_NUM
            
            if tid == TILE_DIM
                Lap[i,j] = grid[tid + 1, 2]
            else
                Lap[i,j] = (c^2/Δx) * grid[tid + 2, 2] + grid[tid, 2] +
                    grid[tid + 1, 3] + grid[tid + 1, 1] - 4grid[tid + 1, 2]
            end
            # if on interior
        else
            Lap[i,j] = (c^2/Δx) * grid[tid + 2, 2] + grid[tid, 2] +
                grid[tid + 1, 3] + grid[tid + 1, 1] - 4grid[tid + 1, 2]
        end
        
    end
    
    return nothing
end

