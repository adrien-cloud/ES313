### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ f85fbaa0-661c-11f0-3381-7f4e2e5abf87
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	while !isfile("Project.toml") && !isdir("Project.toml")
        cd("..")
    end
    Pkg.activate(pwd())
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ ca20668e-3867-4d42-8259-44ce294c11e8
begin
	using Random
	using Agents
	using DataFrames
	using CairoMakie
	using OSMMakie
end

# ╔═╡ 5840a5de-1619-403a-be7f-464b2ccbfb9f
let	
	# make the space the agents will live in
	space = GridSpace((20, 20)) # 20×20 grid cells
	
	# make an agent type appropriate to this space and with the
	# properties we want based on the ABM we will simulate
	@agent struct Schelling(GridAgent{2}) # inherit all properties of `GridAgent{2}`
	    mood::Bool = false # all agents are sad by default :'(
	    group::Int # the group does not have a default value!
	end
	
	# define the evolution rule: a function that acts once per step on
	# all activated agents (acts in-place on the given agent)
	function schelling_step!(agent, model)
	    # Here we access a model-level property `min_to_be_happy`
	    # This will have an assigned value once we create the model
	    minhappy = model.min_to_be_happy
	    count_neighbors_same_group = 0
	    # For each neighbor, get group and compare to current agent's group
	    # and increment `count_neighbors_same_group` as appropriately.
	    # Here `nearby_agents` (with default arguments) will provide an iterator
	    # over the nearby agents one grid cell away, which are at most 8.
	    for neighbor in nearby_agents(agent, model)
	        if agent.group == neighbor.group
	            count_neighbors_same_group += 1
	        end
	    end
	    # After counting the neighbors, decide whether or not to move the agent.
	    # If `count_neighbors_same_group` is at least min_to_be_happy, set the
	    # mood to true. Otherwise, move the agent to a random position, and set
	    # mood to false.
	    if count_neighbors_same_group ≥ minhappy
	        agent.mood = true
	    else
	        agent.mood = false
	        move_agent_single!(agent, model)
	    end
	    return
	end
	
	# make a container for model-level properties
	properties = Dict(:min_to_be_happy => 3)
	
	# Create the central `AgentBasedModel` that stores all simulation information
	model = StandardABM(
	    Schelling, # type of agents
	    space; # space they live in
	    agent_step! = schelling_step!, properties
	)
	
	# populate the model with agents by automatically creating and adding them
	# to random position in the space
	for n in 1:300
	    add_agent_single!(model; group = n < 300 / 2 ? 1 : 2)
	end
	
	# run the model for 5 steps, and collect data.
	# The data to collect are given as a vector of tuples: 1st element of tuple is
	# what property, or what function of agent -> data, to collect. 2nd element
	# is how to aggregate the collected property over all agents in the simulation
	using Statistics: mean
	xpos(agent) = agent.pos[1]
	adata = [(:mood, sum), (xpos, mean)]
	adf, mdf = run!(model, 5; adata)
	adf # a Julia `DataFrame`
end

# ╔═╡ 55810ad4-430c-4a01-823c-13fec55dba0f
html"""
 <! -- this adapts the width of the cells to display its being used on -->
<style>
	main {
		margin: 0 auto;
		max-width: 2000px;
    	padding-left: max(160px, 10%);
    	padding-right: max(160px, 10%);
	}
</style>
"""

# ╔═╡ 9a8f734d-5225-4dda-a44e-96960418ac73
md"""
# Agent-based modelling

Agent-based modelling can easily become complex, even though the basic principles remain unchanged. To introduce you to some applications of ABM, we will delve deeper into the [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) package.

![](https://juliadynamics.github.io/Agents.jl/stable/assets/logo.png)

The main goal will be to go over two examples. First and foremost the tutorial where all steps of a typical workflow are detailed. Secondly, we will cover a more advanced example where we slightly modify one of the cases foreseen in the documentation.

##  Tutorial

To get you up to speed with the typical structure of the Agents.jl package, we first recommend going through the [Tutorial](https://juliadynamics.github.io/Agents.jl/stable/tutorial/). All steps in a typical design process are meticulously explained.

!!! tip "Steps"
    1. Choose kind of space for the agents to live in.
    2. Define the agent type(s).
    3. Define the evolution rule(s).
    4. Initialize the ABM instance.
    5. Evolve the model in time.
    6. Visualize the model.
    7. Collect relevant data.

For your convenience, the tutorial code has already been inserted below.
"""

# ╔═╡ 618fa988-f5d0-4cf0-8068-82d4b6cc920a
md"""
## RMA Escape

For the second application, we will cover a more realistic scenario. We will build our scenario on top of the [Zombie Outbreak in a City](https://juliadynamics.github.io/Agents.jl/stable/examples/zombies/) example. In a first step, go over the example and its implementation. Notice how we are now working in an *OSMSpace* with an *OSMAgent*.

Given this framework, implement following scenario:

!!! info "RMA Outbreak"
	The first year students at the RMA really need to let of some steam. However, the RMA staff does not wish to grant them an additional sortie. Brave as they are, the students collectively decide to risk it and go out in Brussels. The staff, having an inside (wo)man, have become aware of this ambitious escape plan and decide to spread out across Brussels to catch the students in the act.

	We want to know, after an evening of partying, how many students have been caught by a member of staff and will therefore receive a C2. Model this by applying following criteria:

	* Download a map of Brussels using OSM (Open Street Maps). Choose the area yourselves, yet do not make it too large (for performance purposes).
	* Initialize the students and staff randomly across the map.
	* The students roam around Brussels randomly. If they become stationary, there is a 5% chance of them moving elsewhere. Given how they are in good shape, they have a random speed between 5 and 15 km/h.
	* The staff follow a similar behaviour. However, they never become stationary. They keep on roaming the streets of Brussels. Given their old age and higher BMI, their speed is random between 2 and 9 km/h.
	* A student is 'caught' if a staff member moves within 50 m of the student.
	* Experiment with the amount of students (roughly 100) and the amount of staff (roughly 5).
	* Make a video of the interactions.
	* Count how many people will receive a C2.

!!! warning "Installation"
	If you find that it is impossible to install the Makie, CairoMakie and OSMMakie package (on Mac), try these steps:

	Remove the packages (Pkg.rm('Makie'), ...)

	In a terminal window, run:
	```
	sudo chown -R $USER ~/.config/
	sudo chmod -R u+rw ~/.config/
	```


	Enter your password when promted. These steps will make sure that the installer has user level permissions.

	Now reinstall the packages using Pkg.add('Makie'), ... and (if necessary) Pkg.build('Makie')
"""

# ╔═╡ Cell order:
# ╟─55810ad4-430c-4a01-823c-13fec55dba0f
# ╟─f85fbaa0-661c-11f0-3381-7f4e2e5abf87
# ╠═ca20668e-3867-4d42-8259-44ce294c11e8
# ╟─9a8f734d-5225-4dda-a44e-96960418ac73
# ╠═5840a5de-1619-403a-be7f-464b2ccbfb9f
# ╟─618fa988-f5d0-4cf0-8068-82d4b6cc920a
