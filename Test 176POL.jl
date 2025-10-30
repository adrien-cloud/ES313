### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 3f86d540-aeb5-11f0-3937-77b7506ab76d
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	while !isfile("Project.toml") && !isdir("Project.toml")
        cd("..")
    end
    Pkg.activate(pwd())
    
	using Distributions, LinearAlgebra, InteractiveUtils
	using PlutoUI
	using Plots
	using FileIO
	using Graphs
	PlutoUI.TableOfContents()
end

# ╔═╡ 723aae3d-f8ab-42d8-9e55-b04ff4645c14
html"""
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 35c960c3-25a9-4606-99f3-c68e0898e617
md"""
# Test 176POL
"""

# ╔═╡ af661db5-959a-4f81-acbe-35a8a58deb9f
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

# ╔═╡ 60de15c2-7780-4b6a-95c3-201e578a7bbb
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

# ╔═╡ 912c179e-d5b2-434a-967b-082c33cf6962
function evolve_world!(W,state,parameters)
	for (pos,current_state) in vcat([(pos, cs) for (pos, cs) in state if cs.pred],[(pos, cs) for (pos, cs) in state if !cs.pred])
		if current_state != state[pos]
			continue
		end
		prey_neigh=[k for k in Graphs.neighbors(W,pos) if k in keys(state) && !state[k].pred]
        empty_neigh = [k for k in Graphs.neighbors(W, pos) if !(k in keys(state))]

		energy=current_state.E-parameters.E_iter
		pop!(state,pos)
		if current_state.E <= 0
			#@show "dying"
			continue
		end
		
        if !isempty(prey_neigh) && current_state.pred
            target = rand(prey_neigh)
			pop!(state,target)
			energy+=parameters.E_food
			#@show "eating"
        elseif !isempty(empty_neigh)
            target = rand(empty_neigh)
        else
			target = pos
        end

		if current_state.age >= (current_state.pred ? parameters.t_hunter : parameters.t_prey) && target != pos
			state[pos]=(pred=current_state.pred,E=parameters.E_start,age=0)
			state[target]=(pred=current_state.pred,E=energy,age=0)
		else
			state[target]=(pred=current_state.pred,E=energy,age=current_state.age+1)
		end
	end
end

# ╔═╡ f0319dc8-fda7-4144-bee5-70d8e6ec0758
let 
	iter=1000
	state = Dict{Int64, NamedTuple{(:pred, :E, :age), Tuple{Bool, Int64, Int64}}}()
	parameters=(t_prey=5,t_hunter=20,E_start=15,E_iter=1,E_food=5)
	for _ in 1:35  state[rand(1:10000)]=(pred=true,E=parameters.E_start,age=0) end
	for _ in 1:20  state[rand(1:10000)]=(pred=false,E=parameters.E_start,age=0) end
	W=generate_world(100,100)
	hunters=Vector{Int64}()
	preys=Vector{Int64}()

	for i in 1:iter
		push!(hunters,count([cs.pred for (pos, cs) in state]))
		push!(preys,count([!(cs.pred) for (pos, cs) in state]))
		evolve_world!(W,state,parameters)
		if hunters[end] == 0 && preys[end] == 0
			@show i
			break
		end
	end
	
	plot(1:length(hunters),[hunters preys])
end

# ╔═╡ Cell order:
# ╠═3f86d540-aeb5-11f0-3937-77b7506ab76d
# ╠═723aae3d-f8ab-42d8-9e55-b04ff4645c14
# ╟─35c960c3-25a9-4606-99f3-c68e0898e617
# ╠═60de15c2-7780-4b6a-95c3-201e578a7bbb
# ╠═af661db5-959a-4f81-acbe-35a8a58deb9f
# ╠═f0319dc8-fda7-4144-bee5-70d8e6ec0758
# ╠═912c179e-d5b2-434a-967b-082c33cf6962
