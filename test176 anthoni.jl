### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ ce733b10-ace2-11f0-1826-b7bb8196149a
# Explicit use of own environment instead of a local one for each notebook
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	while !isfile("Project.toml") && !isdir("Project.toml")
        cd("..")
    end
    Pkg.activate(pwd())
	using PlutoUI
	PlutoUI.TableOfContents(depth=4)
end

# ╔═╡ dfdba31c-bdf4-44bf-b3e9-55ee4805e14b
begin
	
using NativeSVG # SVG plotting library
using Random    # for random related activities
using Printf    # for nice string plotting
using Distributions, LinearAlgebra, InteractiveUtils
using Plots
using FileIO
using NativeSVG # SVG plotting library
using ImageMagick
using Graphs
using SimpleWeightedGraphs
using GraphPlot
using Statistics
	
	
end

# ╔═╡ 315b5d19-dac6-45ff-bb37-cef0ee1511b0
md"""
# Test 176
"""

# ╔═╡ 5069799d-da10-405c-bd51-3ab4eb015b94
md"""
## Functies gegeven op de test
"""

# ╔═╡ 3efe9f12-fca3-40d0-b282-471d40eecb6a
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

# ╔═╡ 2158132a-92e7-4e17-b136-bedb7bd37bdf
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

# ╔═╡ 929aae4f-1346-4c23-8b18-87bdda868b35
W=generate_world(3,4)

# ╔═╡ ada2ff84-4066-4281-b75d-c154f14d68ca
Graphs.neighbors(W,1)

# ╔═╡ 5be9ebe7-6978-4e17-aa67-104b557bebe3
md"""
### van node nummer naar rij en kolom nummer
"""



# ╔═╡ 9ba5aee9-553c-49ad-a1ac-453bd95dd176
# ╠═╡ disabled = true
#=╠═╡

begin
col = ceil(Int, node_number / rows)
row = node_number - (col - 1) * rows
end 

  ╠═╡ =#

# ╔═╡ 9f17f60f-b6ee-45af-a679-a94d7345fd73
md"""
### van rij en kolom nummer naar node nummer
"""


# ╔═╡ 0ace9b4d-cf09-4d9b-8467-7a520a6eb8a0
# ╠═╡ disabled = true
#=╠═╡
node_number = (col - 1) * rows + row
  ╠═╡ =#

# ╔═╡ 48bb44ce-21a9-498d-b217-6ed58b0457ae
md"""
## Begin oefening
"""

# ╔═╡ 46555825-69e9-44dd-9efd-3b080a6323b4
abstract type Animal end

# ╔═╡ 2050cb58-c438-4b72-88c2-fc44293b4e3d
struct Prey <: Animal
    age::Int
    reproduce_time::Int # 5 jaar
end

# ╔═╡ 41f3390c-d2c1-4849-ac59-f6b77aa8893e
struct Predator <: Animal
    age::Int
    reproduce_time::Int # 10 jaar
    energy::Float64 # E_init = 10, E_loss = 1, E_gain = 5
end

# ╔═╡ 8248e9f8-86b5-4166-8566-2b00ae1e0d03
struct World
    grid::Matrix{Union{Nothing, Animal}}
    nrows::Int
    ncols::Int
    g::Graph
end

# ╔═╡ 1d1e9ee7-728a-48e5-93d9-026e9ab7ad5d
begin

# Hulpfunctie: converteer (row, col) naar node_number
rowcol_to_node(row, col, nrows) = (col - 1) * nrows + row

# Hulpfunctie: converteer node_number naar (row, col)
node_to_rowcol(node, nrows) = ((node - 1) % nrows + 1, ceil(Int, node / nrows))

# Functie die de coördinaten van alle buren geeft
function neighbors_coord(world::World, row, col)
    node = rowcol_to_node(row, col, world.nrows)
    nb_nodes = Graphs.neighbors(world.g, node)
    [node_to_rowcol(nb, world.nrows) for nb in nb_nodes]
end

end 

# ╔═╡ f1566fb2-6034-4b68-b4c3-c870806b194a
"""
	create_world(nrows, ncols; prey_density=0.2, predator_density=0.05)

	functie die de initiële toestand beschrjift, die de eerste dieren op hun plaats zet in het grid
"""
function create_world(nrows, ncols; prey_density=0.2, predator_density=0.05)
    grid = Matrix{Union{Nothing, Animal}}(nothing, nrows, ncols)
    g = generate_world(nrows, ncols)
    for y in 1:nrows, x in 1:ncols
        r = rand()
        if r < prey_density
            grid[y, x] = Prey(0, 5)
        elseif r < prey_density + predator_density
            grid[y, x] = Predator(0, 10, 10.0)
        end
    end
    return World(grid, nrows, ncols, g)
end

# ╔═╡ 1015fa2f-0646-4ecd-abbf-c8b44df4df93
function empty_neighbors(world::World, row, col)
    coords = neighbors_coord(world, row, col)
    [c for c in coords if world.grid[c...] === nothing]
end

# ╔═╡ 1911d0d0-82fa-4419-b0e6-ade91ec315c0
function prey_neighbors(world::World, row, col)
    coords = neighbors_coord(world, row, col)
    [c for c in coords if world.grid[c...] isa Prey]
end

# ╔═╡ afdd4306-11a8-444e-a640-5b886fce5ee2
# ╠═╡ disabled = true
#=╠═╡
function move_prey!(world::World)
    nrows, ncols = world.nrows, world.ncols
    newgrid = copy(world.grid)

    for y in 1:nrows, x in 1:ncols
        animal = world.grid[y, x]
        if animal isa Prey # isa gebruiken om type van het object te controlleren
            a = animal::Prey
            a = Prey(a.age + 1, a.reproduce_time)
            free = empty_neighbors(world, y, x)
           
			if !isempty(free)
			    newpos = rand(free)
			    if a.age >= a.reproduce_time
			        # voortplanting
			        newgrid[newpos...] = Prey(0, a.reproduce_time)
			        newgrid[y, x] = Prey(0, a.reproduce_time)
			    else
			        newgrid[newpos...] = a
			        newgrid[y, x] = nothing
			    end
			else
			    newgrid[y, x] = a  # blijf staan
			end
		end 
    end
    world.grid[:,:] = newgrid[:,:]
	#return world.grid
end
  ╠═╡ =#

# ╔═╡ 5bf31c14-28dc-4edd-892b-e5c4fb9d14c9
# ╠═╡ disabled = true
#=╠═╡
function move_predator!(world::World; energy_loss=1.0, energy_gain=5.0)
    nrows, ncols = world.nrows, world.ncols
    newgrid = copy(world.grid)

    for y in 1:nrows, x in 1:ncols
        animal = world.grid[y, x]
        if animal isa Predator
            p = animal::Predator
            # Leeftijd en energie updaten
            p = Predator(p.age + 1, p.reproduce_time, p.energy - energy_loss)

            # Sterfte door energieverlies
            if p.energy <= 0
                newgrid[y, x] = nothing
                continue
            end

            prey_neigh = prey_neighbors(world, y, x)
            free = empty_neighbors(world, y, x)

            if !isempty(prey_neigh)
                # Eet een willekeurige prooi
                target = rand(prey_neigh)
                p = Predator(p.age, p.reproduce_time, p.energy + energy_gain)
                # Plaats roofdier op prooi-locatie
                newgrid[target...] = p
                # Voortplanting check
                if p.age >= p.reproduce_time
                    newgrid[y, x] = Predator(0, p.reproduce_time, 10.0)
                else
                    newgrid[y, x] = nothing
                end

            elseif !isempty(free)
                # Verplaats naar vrije cel
                target = rand(free)
                newgrid[target...] = p
                if p.age >= p.reproduce_time
                    # Plaats nieuw roofdier op originele plek
                    newgrid[y, x] = Predator(0, p.reproduce_time, 10.0)
                else
                    newgrid[y, x] = nothing
                end

            else
                # Blijf op plaats
                newgrid[y, x] = p
            end
        end
    end

    world.grid[:,:] = newgrid[:,:]
end


  ╠═╡ =#

# ╔═╡ 66cdb728-98ac-48c8-b820-027d7b212992
function move_prey!(world::World)
    nrows, ncols = world.nrows, world.ncols
    newgrid = copy(world.grid)
    moved = falses(nrows, ncols)  # houdt bij welke dieren al zijn verplaatst

    # verzamel alle coördinaten van prooien en shuffle
    coords = [(y, x) for y in 1:nrows, x in 1:ncols if world.grid[y,x] isa Prey]
    shuffle!(coords)

    for (y, x) in coords
        # als dit dier al verplaatst is, overslaan
        if moved[y, x]
            continue
        end

        animal = world.grid[y, x]::Prey
        a = Prey(animal.age + 1, animal.reproduce_time)

        # vind vrije naburige cellen die nog niet zijn gekozen door andere dieren
        free = [c for c in empty_neighbors(world, y, x) if !moved[c...]]
        
        if !isempty(free)
            newpos = rand(free)
            if a.age >= a.reproduce_time
                # voortplanting: ouder blijft, nieuwe prooi naar vrije cel
                newgrid[y, x] = Prey(0, a.reproduce_time)
                newgrid[newpos...] = Prey(0, a.reproduce_time)
                moved[y, x] = true
                moved[newpos...] = true
            else
                # verplaatsing
                newgrid[newpos...] = a
                newgrid[y, x] = nothing
                moved[newpos...] = true
            end
        else
            # geen vrije buren, blijf op plaats
            newgrid[y, x] = a
            moved[y, x] = true
        end
    end

    world.grid[:,:] = newgrid[:,:]
end


# ╔═╡ 1c1caf15-80d6-41b9-8c2e-78363e4af578
function move_predator!(world::World; energy_loss=1.0, energy_gain=3.0)
    nrows, ncols = world.nrows, world.ncols
    newgrid = copy(world.grid)
    moved = falses(nrows, ncols)  # houdt bij welke roofdieren al actie hebben uitgevoerd

    # verzamel alle coördinaten van roofdieren en shuffle
    coords = [(y, x) for y in 1:nrows, x in 1:ncols if world.grid[y,x] isa Predator]
    shuffle!(coords)

    for (y, x) in coords
        # als dit roofdier al verplaatst is, overslaan
        if moved[y, x]
            continue
        end

        animal = world.grid[y, x]::Predator
        p = Predator(animal.age + 1, animal.reproduce_time, animal.energy - energy_loss)

        # sterf als energie op is
        if p.energy <= 0
            newgrid[y, x] = nothing
            moved[y, x] = false
            continue
        end

        # naburige prooien en vrije cellen die nog niet gebruikt zijn
        prey_neigh = [c for c in prey_neighbors(world, y, x) if !moved[c...]]
        free = [c for c in empty_neighbors(world, y, x) if !moved[c...]]

        if !isempty(prey_neigh)
            # eet een willekeurige prooi
            target = rand(prey_neigh)
			 	if p.age >= p.reproduce_time
					newgrid[target...] = Predator(0, p.reproduce_time, p.energy + energy_gain)
				else
					newgrid[target...] = Predator(p.age, p.reproduce_time, p.energy + energy_gain)
				end 
			
            moved[target...] = true

            # voortplanting
            if p.age >= p.reproduce_time
                newgrid[y, x] = Predator(0, p.reproduce_time, 10.0)
                moved[y, x] = true
            else
                newgrid[y, x] = nothing
                moved[y, x] = false
            end

        elseif !isempty(free)
            # verplaats naar vrije cel
            target = rand(free)
			
			if p.age >= p.reproduce_time
					newgrid[target...] = Predator(0, p.reproduce_time, p.energy)
				else
					newgrid[target...] = Predator(p.age, p.reproduce_time, p.energy)
				end 
			
            moved[target...] = true

            # voortplanting
            if p.age >= p.reproduce_time
				p = Predator(0, p.reproduce_time, p.energy)
                newgrid[y, x] = Predator(0, p.reproduce_time, 10.0)
                moved[y, x] = true
            else
                newgrid[y, x] = nothing
                moved[y, x] = false
            end

        else
            # geen verplaatsing mogelijk, blijf op plaats
            newgrid[y, x] = p
            moved[y, x] = true
        end
    end

    world.grid[:,:] = newgrid[:,:]
end


# ╔═╡ 6cdc7e72-37ca-4c30-9bdf-0060240bf313
# ╠═╡ disabled = true
#=╠═╡
function step!(world::World,x::Int,y::Int;energy_loss=1.0, energy_gain=5.0)
	nrows, ncols = world.nrows, world.ncols

	prey_neigh = [c for c in prey_neighbors(world, y, x)]
	free_neigh = [c for c in empty_neighbors(world, y, x)]

	if world.grid[y,x] isa Predator

		if !isempty(prey_neigh)
			
		
    	animal = world.grid[y, x]::Predator
        p = Predator(animal.age + 1, animal.reproduce_time, animal.energy- energy_loss)

		

		
		
	

  ╠═╡ =#

# ╔═╡ 14935a53-8b19-468e-9909-054dd179b2b0
function world_to_matrix(world::World)
    mat = zeros(Int, world.nrows, world.ncols)
    for i in 1:world.nrows, j in 1:world.ncols
        if world.grid[i,j] isa Prey
            mat[i,j] = 1
        elseif world.grid[i,j] isa Predator
            mat[i,j] = 2
        end
    end
    return mat
end


# ╔═╡ 2e289d81-7698-429b-85cf-37631ca6fc8b
function simulate!(world::World, nsteps::Int)
	
	nrows, ncols = world.nrows, world.ncols
	geschiedenis = zeros(Int,nsteps+1, nrows, ncols)

	geschiedenis[1,:,:] = world_to_matrix(world)
	
    for step in 2:nsteps+1
        move_predator!(world)
        move_prey!(world)
		geschiedenis[step,:,:] = world_to_matrix(world)
    end

	return geschiedenis
end


# ╔═╡ 90a58c5a-6523-41a0-9485-987486276feb
world_test = create_world(10,10;prey_density=0.2, predator_density=0.05)

# ╔═╡ 7eadfc48-eb65-4132-b70b-72eb43a82786
geschiedenis1 = simulate!(world_test, 50);

# ╔═╡ 42f4538f-b72f-4c7a-b5af-0d9a148027b6
@bind tijd_balkje Slider(1:51, show_value=true)

# ╔═╡ 838a74e1-aa98-4efc-8982-566cc7bb6de5
heatmap(
        reverse(geschiedenis1[tijd_balkje,:,:],dims=1),
        c = [:white, :brown, :black],  # 0=leeg, 1=boom, 2=brand
        clims = (0, 2),              # kleurenschaal fixeren
        title = "Lotka Volterra",
        size = (500,500),
        axis = nothing,              # geen assen
        aspect_ratio = 1,            # vierkant beeld
        framestyle = :none,          # geen randlijnen
        legend = false  )             # geen kleurenbalk


# ╔═╡ ef03fc3d-ce9c-4d0b-971d-4d4fdd2abce9
function plot_aantal(nrows, ncols, nsteps::Int)

	world_test = create_world(nrows,ncols;prey_density=0.1, predator_density=0.05)
	geschiedenis = simulate!(world_test, nsteps)

	
	prooi = zeros(nsteps+1)
	jager = zeros(nsteps+1)

	for i in 1:nsteps+1

		prooi[i] = sum(geschiedenis[i,:,:].== 1)
		jager[i] = sum(geschiedenis[i,:,:].==2)

	end 

	tijd = 1:nsteps+1;

	plot(tijd, prooi, label="prooi", lw=2, color=:brown)
	plot!(tijd, jager, label="jager", lw=2, color=:black)
	xlabel!("jaren")
	ylabel!("Aantal dieren")
	title!("Lotka-Volterra")
end 

# ╔═╡ 53e9ba61-4c74-4558-ad99-99cd8135a5a3
plot_aantal(10,10,50)

# ╔═╡ Cell order:
# ╠═ce733b10-ace2-11f0-1826-b7bb8196149a
# ╠═dfdba31c-bdf4-44bf-b3e9-55ee4805e14b
# ╟─315b5d19-dac6-45ff-bb37-cef0ee1511b0
# ╠═5069799d-da10-405c-bd51-3ab4eb015b94
# ╠═2158132a-92e7-4e17-b136-bedb7bd37bdf
# ╠═3efe9f12-fca3-40d0-b282-471d40eecb6a
# ╠═929aae4f-1346-4c23-8b18-87bdda868b35
# ╠═ada2ff84-4066-4281-b75d-c154f14d68ca
# ╟─5be9ebe7-6978-4e17-aa67-104b557bebe3
# ╠═9ba5aee9-553c-49ad-a1ac-453bd95dd176
# ╟─9f17f60f-b6ee-45af-a679-a94d7345fd73
# ╠═0ace9b4d-cf09-4d9b-8467-7a520a6eb8a0
# ╠═1d1e9ee7-728a-48e5-93d9-026e9ab7ad5d
# ╠═48bb44ce-21a9-498d-b217-6ed58b0457ae
# ╠═46555825-69e9-44dd-9efd-3b080a6323b4
# ╠═2050cb58-c438-4b72-88c2-fc44293b4e3d
# ╠═41f3390c-d2c1-4849-ac59-f6b77aa8893e
# ╠═8248e9f8-86b5-4166-8566-2b00ae1e0d03
# ╠═f1566fb2-6034-4b68-b4c3-c870806b194a
# ╠═1015fa2f-0646-4ecd-abbf-c8b44df4df93
# ╠═1911d0d0-82fa-4419-b0e6-ade91ec315c0
# ╠═afdd4306-11a8-444e-a640-5b886fce5ee2
# ╠═5bf31c14-28dc-4edd-892b-e5c4fb9d14c9
# ╠═66cdb728-98ac-48c8-b820-027d7b212992
# ╠═1c1caf15-80d6-41b9-8c2e-78363e4af578
# ╠═6cdc7e72-37ca-4c30-9bdf-0060240bf313
# ╠═14935a53-8b19-468e-9909-054dd179b2b0
# ╠═2e289d81-7698-429b-85cf-37631ca6fc8b
# ╠═90a58c5a-6523-41a0-9485-987486276feb
# ╠═7eadfc48-eb65-4132-b70b-72eb43a82786
# ╠═42f4538f-b72f-4c7a-b5af-0d9a148027b6
# ╠═838a74e1-aa98-4efc-8982-566cc7bb6de5
# ╠═ef03fc3d-ce9c-4d0b-971d-4d4fdd2abce9
# ╠═53e9ba61-4c74-4558-ad99-99cd8135a5a3
