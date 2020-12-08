using CUDA



function lap_1d_shared_kern!(U, Lap, c, Δx, n, ::Val{TILE_DIM}, ::Val{BLOCK_NUM}) where {TILE_DIM, BLOCK_NUM}

    tid = threadIdx().x
    bid = blockIdx().x

    i = (bid - 1) * TILE_DIM + tid

    # initialize shared memory
    if bid == 1 || bid == BLOCK_NUM
        grid = @cuStaticSharedMem(Float64, (TILE_DIM + 1, 3))
    else
        grid = @cuStaticSharedMem(Float64, (TILE_DIM + 1, 3))
    end
    
    for j in 2:3

        
        #############################
        #      load shared memory   #
        #############################
        
        # north boundary
        if bid == 1
            # halo node
            if tid == TILE_DIM
                @cuprintln("first block halo looking at $i and $(i+1)")
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
                @cuprintln("last block halo Looking at $i and $(i-1)")
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
                @cuprintln("shared memory is $size(grid)[1] and $size(grid)[2]")
                grid[tid + 1, 1] = U[i,j-1]
                grid[tid + 1, 2] = U[i,j]
                grid[tid + 1, 3] = U[i,j+1]
                # halo outside
                grid[tid + 1, 1] = U[i+1,j-1]
                grid[tid + 1, 2] = U[i+1,j]
                grid[tid + 1, 3] = U[i+1,j+1]

            end
        end
    end
    #=
            else
                # load my indices
                grid[tid + 1, 1] = U[i,j-1]
                grid[tid + 1, 2] = U[i,j]
                grid[tid + 1, 3] = U[i,j+1]
            end
        end
    end
            
        ##################################
        #   Compute with shared memory   # 
        ##################################

            # if on north boundary
            if bid == 1
                if tid == 1
                    Lap[i,j] = grid[tid, 2]
                else
                    Lap[i,j] = grid[tid + 1, 2] + grid[tid - 1, 2] +
                        grid[tid, 3] + grid[tid, 1] - 4grid[tid, 2]
                end

            # if on south boundary
            elseif bid == BLOCK_NUM
                
                if tid == TILE_DIM
                    Lap[i,j] = grid[tid + 1, 2]

                else
                    Lap[i,j] = grid[tid + 2, 2] + grid[tid, 2] +
                        grid[tid + 1, 3] + grid[tid + 1, 1] - 4grid[tid + 1, 2]
                end

            # if on interior
            else
                Lap[i,j] = grid[tid + 2, 2] + grid[tid, 2] +
                    grid[tid + 1, 3] + grid[tid + 1, 1] - 4grid[tid + 1, 2]
            end
        end
        
    end
    =#
    return nothing
end
