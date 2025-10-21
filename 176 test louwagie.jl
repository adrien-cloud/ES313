using Pkg
# this is redundant if you run it through start.jl, but to make sure...
while !isfile("Project.toml") && !isdir("Project.toml")
    cd("..")
end
Pkg.activate(pwd())

using Graphs
using Plots

"""
generate_world(n::Int, m::Int)
Generates a graph that corresponds with the projection of a torus onto a 2-D space.

# Arguments
- N{Int}: the number of discrete points along the major circumference of the torus.
- M{Int}: the number of discrete points along the minor circumference of the torus.

*Note*: the node labeling is columnwise, so a 2x3 graph has the following layout
```
1 3 5
2 4 6
```
and node 4 will be connected to nodes 1, 2, 3, 4 and 6.

In a large graph, nodes will always have 8 unique neighbors:
```
1 4 7 10
2 5 8 11
3 6 9 12
```
```julia
W = generate_world(3, 4)
neighbors(W, 1) # returns [2, 3, 4, 5, 6, 10, 11, 12]
````
"""
function generate_world(n::Int, m::Int)
    # generate a periodic grid graph (only contains horizontal and vertical connections)
    g = Graphs.grid([n,m], periodic=true)
    # add the diagonal connections between the nodes
    for i in 1:nv(g)
        for diag_neighbor in get_toroidal_diagonal_neighbors(i, n, m)
            add_edge!(g, i, diag_neighbor)
        end
    end
    # return the graph
    return g
end

"""
get_toroidal_diagonal_neighbors(node_number, rows, cols)
Returns the node numbers of the diagonal neighbors of the `node_number` in a toroidal grid of size (`rows`, `cols`)
using column major indexing.

# Example
```julia
rows, cols = 2, 4
# Get neighbors for node 2
neighbors_of_2 = get_toroidal_neighbors(2, rows, cols)
println("Neighbors of 2: ", neighbors_of_2)
# Get neighbors for node 4
neighbors_of_4 = get_toroidal_neighbors(4, rows, cols)
println("Neighbors of 4: ", neighbors_of_4)
"""
function get_toroidal_diagonal_neighbors(node_number, rows, cols)
    # Calculate the current row and column based on the node number
    col = ceil(Int, node_number / rows)
    row = node_number - (col - 1) * rows

    # Initialise
    neighbors = Set{Int}()

    # Define the relative positions of all diagonal neighbors
    diagonal_neighbor_positions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Iterate through all possible neighbors
    for (dr, dc) in diagonal_neighbor_positions
        # Calculate the neighbor's row and column indices
        # Use modulo operation to wrap around the toroidal grid
        # Adding rows and cols before modulo ensures positive indices
        neighbor_row = mod1(row + dr, rows)
        neighbor_col = mod1(col + dc, cols)

        # Convert back to node number
        neighbor_number = (neighbor_col - 1) * rows + neighbor_row

        # Append the neighbor to the set
        push!(neighbors, neighbor_number)
    end
    return neighbors
end


mutable struct prey
    node_number::Int64
    age::Int64
end

mutable struct predator
    node_number::Int64
    age::Int64
    energy::Int64
end
predator(node_number::Int64, age::Int64) = predator(node_number, age, E_start)

global (rows, cols) = (100,100)
global t_prey = 1  
global t_predator = 3  
global E_start = 7    
global E_gain_prey = 4 
global world_map = generate_world(rows, cols)
global world_matrix = Matrix{Union{prey, predator, Nothing}}(fill(nothing, rows, cols))
global iterations = 100



function new_node_number(agent::prey)
    node_number = agent.node_number
    neighbors_list = neighbors(world_map, node_number)
    new_node_number = rand(neighbors_list)

    prey_neighbors = Int[]
    predator_neighbors = Int[]

    for i in neighbors_list
        if typeof(world_matrix[matrix_coord(i)...]) == prey
            push!(prey_neighbors, i)
        elseif typeof(world_matrix[matrix_coord(i)...]) == predator
            push!(predator_neighbors, i)
        end
    end
    neighbors_list = setdiff(neighbors_list, prey_neighbors)
    neighbors_list = setdiff(neighbors_list, predator_neighbors)
    if !isempty(neighbors_list)
        new_node_number = rand(neighbors_list)
    else
        new_node_number = agent.node_number
    end
    return new_node_number
end

function new_node_number(agent::predator)
    node_number = agent.node_number
    neighbors_list = neighbors(world_map, node_number)
    
    prey_neighbors = Int[]
    predator_neighbors = Int[]

    for i in neighbors_list  
        if typeof(world_matrix[matrix_coord(i)...]) == prey
            push!(prey_neighbors, i)
        elseif typeof(world_matrix[matrix_coord(i)...]) == predator
            push!(predator_neighbors, i)
        end
    end

    neighbors_list = setdiff(neighbors_list, predator_neighbors)
    if !isempty(prey_neighbors)
        new_node_number = rand(prey_neighbors)
    elseif !isempty(neighbors_list)
        new_node_number = rand(neighbors_list)
    else
        new_node_number = agent.node_number
    end

    return new_node_number
end

function step_agent!(agent::Union{prey, predator})
    world_matrix[matrix_coord(agent.node_number)...] = nothing
    birth_agent(agent)

    agent.node_number = new_node_number(agent)

    if typeof(agent) == predator
        agent.energy -=1
        if typeof(world_matrix[matrix_coord(agent.node_number)...]) == prey
            agent.energy += E_gain_prey
        end
        if agent.energy <=0
            world_matrix[matrix_coord(agent.node_number)...] = nothing
        else
            world_matrix[matrix_coord(agent.node_number)...] = agent
        end
    else
        world_matrix[matrix_coord(agent.node_number)...] = agent
    end

    agent.age +=1
end

function birth_agent(agent::Union{prey, predator})
    if agent.age >= (typeof(agent) == prey ? t_prey : t_predator)
        world_matrix[matrix_coord(agent.node_number)...] = typeof(agent)(agent.node_number, 0)
        agent.age = 0
    end
end

function matrix_coord(node_number::Int64)
    row = mod1(node_number, rows)
    col = ceil(Int, node_number / rows)
    return (row, col)
end


random_node_number1 = rand(1:rows*cols)
random_node_number2 = rand(1:rows*cols)
while true

    if random_node_number1 != random_node_number2
        world_matrix[matrix_coord(random_node_number1)...] = prey(random_node_number1, 0)
        world_matrix[matrix_coord(random_node_number2)...] = predator(random_node_number2, 0)   
        break
    end
    global random_node_number1 = rand(1:rows*cols)
    global random_node_number2 = rand(1:rows*cols)
end

log = Array{Union{prey, predator, Nothing},3}(undef, iterations, rows, cols)

for step in 1:iterations
    log[step, :, :] = copy(world_matrix)
    #display(world_matrix)
    for i in 1:cols*rows
        if world_matrix[matrix_coord(i)...] != nothing
                step_agent!(world_matrix[matrix_coord(i)...])      
        end
    end    
end

p1 = plot()
for step in 1:iterations
    scatter!([step], [count(x -> typeof(x) == prey, log[step, :, :])], markersize=2, markercolor=:blue, label = nothing)
    scatter!([step], [count(x -> typeof(x) == predator, log[step, :, :])], markersize=2, markercolor=:red, label = nothing)
end


plot!(xlabel="Time step", ylabel="Population", title="Population Dynamics")
# Add a single legend entry for each type
plot!([], [], label="Prey", color=:blue)
plot!([], [], label="Predator", color=:red)
display(p1)