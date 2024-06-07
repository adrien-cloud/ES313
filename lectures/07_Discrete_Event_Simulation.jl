### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 81adcbe6-e0db-4eca-acc1-9a871ad884ff
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	cd(joinpath(dirname(@__FILE__),".."))
    Pkg.activate(pwd())

	# the source file
	include("./lectures/DES_spt.jl")
	txt = readlines("./lectures/DES_spt.jl")
	
	# helper function to find code for a function in the spt file
	function function_source_extractor(src, target)
		start_idx = findfirst(x -> occursin("function $(target)", x), src)
	
		end_idx = findfirst( x -> startswith(x, "end"), @view src[start_idx:end])
	
		#return join(src[start_idx:start_idx+end_idx-1], "\n")
		return join(["```julia", src[start_idx:start_idx+end_idx-1]..., "```"], "\n")
	end

	function struct_source_extractor(src, target)
		start_idx = findfirst(x -> occursin("struct $(target)", x), src)
	
		end_idx = findfirst( x -> startswith(x, "end"), @view src[start_idx:end])
	
		#return join(src[start_idx:start_idx+end_idx-1], "\n")
		return join(["```julia", src[start_idx:start_idx+end_idx-1]..., "```"], "\n")
	end

	
	import PlutoUI: TableOfContents
	TableOfContents(depth=4)
end

# ╔═╡ 60521568-6a00-4c6b-b11a-1d53d75417c2
# dependencies
begin
	using ResumableFunctions
	using ConcurrentSim
	using Logging

	using Distributions
	using Plots
	using StatsPlots
end

# ╔═╡ ef11256b-c5ee-43d4-8757-918a05ad3d1d
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

# ╔═╡ 1546ff80-0faf-11eb-2e03-a3ed4337a0df
md"""# Discrete event simulation with `ConcurrentSim`
## Resumable functions
In Julia, the building block for discrete event simulations are resumable functions. These are made available via the [ResumableFunctions](https://github.com/JuliaDynamics/ResumableFunctions.jl) package. This allow you to create functions that only continue to be evaluated when explicitely called upon.

**Note**: Defining the `@resumable` functions in a notebook can often lead to issues, which is why we rely on a separate file to define the relevant funcions that will be used.

### Example
Consider the fibonacci sequence. We can implement a `@resumable` function that gives us the next number in the sequence (until a predefined end) each time we call it:

$(Markdown.parse(function_source_extractor(txt, "fibonacci")))


"""

# ╔═╡ d3d96c66-de37-49b6-94a5-be888f45f3a3
# example usage
let
	# length of the sequence
	N = 10
	# define the object
	fib = fibonacci(N)
	# call `N` times
	for i in 1:N
		@info fib()
	end
end
	

# ╔═╡ 4d63bf51-df9f-4c7b-8008-5ebf956571fe
md"""

## `ConcurrentSim` basics

[ConcurrentSim](https://github.com/JuliaDynamics/ConcurrentSim.jl) is a discrete-event simulation library. The behavior of active components (e.g. vehicles, customers or messages) is modeled with processes. All processes live in an environment. They interact with the environment and with each other via events.

Processes are described by `@resumable` functions. You can call them process function. During their lifetime, they create events and `@yield` them in order to wait for them to be triggered.

When a process yields an event, the process gets suspended. ConcurrentSim resumes the process, when the event occurs (we say that the event is triggered). Multiple processes can wait for the same event. ConcurrentSim resumes them in the same order in which they yielded that event.

An important event type is the `timeout`. Events of this type are scheduled after a certain amount of (simulated) time has passed. They allow a process to sleep (or hold its state) for the given time. A `timeout` and all other events can be created by calling a constructor having the environment as first argument.
"""

# ╔═╡ 0153c520-0fb0-11eb-1253-0bd7e9dd04ee
md"""### Our First Process

Our first example will be a car process. The car will alternately drive and park for a while. When it starts driving (or parking), it will print the current simulation time.

So let’s start:

$(Markdown.parse(function_source_extractor(txt, "car")))

Our car process requires a reference to an `Environment` in order to create new events. The car‘s behavior is described in an infinite loop. Remember, the car function is a `@resumable function`. Though it will never terminate, it will pass the control flow back to the simulation once a `@yield` statement is reached. Once the yielded event is triggered (“it occurs”), the simulation will resume the function at this statement.

As said before, our car switches between the states parking and driving. It announces its new state by printing a message and the current simulation time (as returned by the function call `now`). It then calls the constructor `timeout` to create a timeout event. This event describes the point in time the car is done parking (or driving, respectively). By yielding the event, it signals the simulation that it wants to wait for the event to occur.

Now that the behavior of our car has been modeled, lets create an instance of it and see how it behaves:

"""

# ╔═╡ 2326d3e2-0fb0-11eb-1827-6d08b4032c34
let sim = Simulation()
	@process car(sim)
	run(sim, 15)
end

# ╔═╡ 31d8458e-0fb0-11eb-0fe5-0b2a1a5d56c3
md"""The first thing we need to do is to create an environment, e.g. an instance of `Simulation`. The macro `@process` having as argument a car process function call creates a process that is initialised and added to the environment automatically.

Note, that at this time, none of the code of our process function is being executed. Its execution is merely scheduled at the current simulation time.

The `Process` returned by the `@process` macro can be used for process interactions.

Finally, we start the simulation by calling `run` and passing an end time to it."""

# ╔═╡ b106b67e-0fb0-11eb-04f2-a734c4fa2e3b
md"""### Process Interaction

The `Process` instance that is returned by `@process` macro can be utilized for process interactions. The two most common examples for this are to wait for another process to finish and to interrupt another process while it is waiting for an event.

#### Waiting for a Process

As it happens, a ConcurrentSim `Process` can be used like an event. If you yield it, you are resumed once the process has finished. Imagine a car-wash simulation where cars enter the car-wash and wait for the washing process to finish, or an airport simulation where passengers have to wait until a security check finishes.

Lets assume that the car from our last example is an electric vehicle. Electric vehicles usually take a lot of time charging their batteries after a trip. They have to wait until their battery is charged before they can start driving again.

We can model this with an additional charge process for our car. Therefore, we redefine our car process function and add a charge process function.

A new charge process is started every time the vehicle starts parking. By yielding the `Process` instance that the `@process` macro returns, the run process starts waiting for it to finish:

$(Markdown.parse(function_source_extractor(txt, "charge")))

$(Markdown.parse(function_source_extractor(txt, "car2")))
"""

# ╔═╡ eda6bb80-0fb0-11eb-2f87-d586f0b69f26
md"""Starting the simulation is straightforward again: We create a `Simulation`, one (or more) cars and finally call `run`."""

# ╔═╡ f67c3cce-0fb0-11eb-3349-4181800dde90
let sim = Simulation()
	@process car2(sim)
	run(sim, 15)
end;

# ╔═╡ 061cc970-0fb1-11eb-0210-d7d0f215be77
md"""#### Interrupting Another Process

Imagine, you don’t want to wait until your electric vehicle is fully charged but want to interrupt the charging process and just start driving instead.

ConcurrentSim allows you to interrupt a running process by calling the `interrupt` function:

$(Markdown.parse(function_source_extractor(txt, "driver")))
"""

# ╔═╡ 32c0db60-0fb1-11eb-3d75-f1a189baf482
md"""The driver process has a reference to the car process. After waiting for 3 time steps, it interrupts that process.

Interrupts are thrown into process functions as `Interrupt` exceptions that can (should) be handled by the interrupted process. The process can then decide what to do next (e.g., continuing to wait for the original event or yielding a new event):

$(Markdown.parse(function_source_extractor(txt, "car3")))
"""

# ╔═╡ f8b8f0ee-0fb1-11eb-13d6-571cad22a19a
md"""When you compare the output of this simulation with the previous example, you’ll notice that the car now starts driving at time 3 instead of 5:"""

# ╔═╡ 05760cb2-0fb2-11eb-2bc4-dd2a2bfde126
let sim = Simulation()
	car_process = @process car3(sim)
	@process driver(sim, car_process)
	run(sim, 15)
end;

# ╔═╡ 1d92d712-0fb2-11eb-23fd-4554371e8a3b
md"""### Shared Resources

ConcurrentSim offers three types of resources that help you modeling problems, where multiple processes want to use a resource of limited capacity (e.g., cars at a fuel station with a limited number of fuel pumps) or classical producer-consumer problems.

In this section, we’ll briefly introduce ConcurrentSim’s Resource class.

#### Basic Resource Usage

We’ll slightly modify our electric vehicle process car that we introduced in the last sections.

The car will now drive to a battery charging station (BCS) and request one of its two charging spots. If both of these spots are currently in use, it waits until one of them becomes available again. It then starts charging its battery and leaves the station afterwards:

$(Markdown.parse(function_source_extractor(txt, "car4")))

The resource’s `request` function generates an event that lets you wait until the resource becomes available again. If you are resumed, you “own” the resource until you release it.

You are responsible to call release once you are done using the resource. When you `release` a resource, the next waiting process is resumed and now “owns” one of the resource’s slots. The basic Resource sorts waiting processes in a FIFO (first in—first out) way.

A resource needs a reference to an `Environment` and a capacity when it is created.

We can now create the car processes and pass a reference to our resource as well as some additional parameters to them. 

Finally, we can start the simulation. Since the car processes all terminate on their own in this simulation, we don’t need to specify an until time — the simulation will automatically stop when there are no more events left:
"""

# ╔═╡ 7845ae80-0fb2-11eb-195b-4d0d4ffca60b
let sim = Simulation(), bcs = ConcurrentSim.Resource(sim, 2)
	for i in 1:4
		@process car4(sim, i, bcs, 2i, 5)
	end
	run(sim)
end

# ╔═╡ 9a77cd30-0fb2-11eb-1196-df634baf199d
md"""#### Priority resource

As you may know from the real world, not every one is equally important. To map that to ConcurrentSim, the methods `request(res, priority=priority)` and `release(res, priority=priority)` lets requesting and releasing processes provide a priority for each request/release. More important requests/releases will gain access to the resource earlier than less important ones. Priority is expressed by integer numbers; smaller numbers mean a higher priority:

$(Markdown.parse(function_source_extractor(txt, "resource_user")))
"""

# ╔═╡ d683334e-0fb2-11eb-3dc4-4be3043674a7
let sim = Simulation(), res = ConcurrentSim.Resource(sim, 1)
	@process resource_user(sim, 1, res, 0.0, 0)
	@process resource_user(sim, 2, res, 1.0, 0)
	@process resource_user(sim, 3, res, 2.0, -1)
	run(sim)
end

# ╔═╡ eff30a90-0fb2-11eb-01c6-c3a35033923d
md"""Although the third process requested the resource later than the second, it could use it earlier because its priority was higher."""

# ╔═╡ 066d6f8e-0fb3-11eb-0979-ab193faf5653
md"""#### Containers

Containers help you modelling the production and consumption of a homogeneous, undifferentiated bulk. It may either be continuous (like water) or discrete (like apples).

You can use this, for example, to model the gas / petrol tank of a gas station. Tankers increase the amount of gasoline in the tank while cars decrease it.

The following example is a very simple model of a gas station with a limited number of fuel dispensers (modeled as `Resource`) and a tank modeled as `Container`:

$(Markdown.parse(struct_source_extractor(txt, "GasStation")))

$(Markdown.parse(function_source_extractor(txt, "monitor_tank")))

$(Markdown.parse(function_source_extractor(txt, "tanker")))

$(Markdown.parse(function_source_extractor(txt, "car5")))

$(Markdown.parse(function_source_extractor(txt, "car_generator")))
"""

# ╔═╡ 13ba2120-0fb3-11eb-0599-831b085542c8
let sim = Simulation()
	gs = GasStation(sim)
	@process car_generator(sim, gs)
	@process monitor_tank(sim, gs)
	run(sim, 55.0)
end

# ╔═╡ 461ec14e-0fb5-11eb-38f0-61ad30ab4e1e
md"""Priorities can be given to a `put` or a `get` event by setting the named argument priority."""

# ╔═╡ 5d04b960-0fb5-11eb-01fe-fd0ff58627f6
md"""#### Stores

Using a `Store` you can model the production and consumption of concrete objects (in contrast to the rather abstract “amount” stored in a Container). A single `Store` can even contain multiple types of objects.

A custom function can also be used to filter the objects you get out of the `store`.

Here is a simple example modelling a generic producer/consumer scenario:

$(Markdown.parse(function_source_extractor(txt, "producer")))

$(Markdown.parse(function_source_extractor(txt, "consumer")))
"""

# ╔═╡ 23885800-0fb4-11eb-0378-1191e7ec5ca8
let sim = Simulation()
	sto = Store{String}(sim, capacity=UInt(2))
	@process producer(sim, sto)
	consumers = [@process consumer(sim, i, sto) for i=1:2]
	run(sim, 5.0)
end

# ╔═╡ 835bf0b0-0fb5-11eb-2a68-275326c4e73c
md"""A `Store` with a filter on the `get` event can, for example, be used to model machine shops where machines have varying attributes. This can be useful if the homogeneous slots of a` Resource` are not what you need:

$(Markdown.parse(struct_source_extractor(txt, "Machine")))

$(Markdown.parse(function_source_extractor(txt, "user")))

$(Markdown.parse(function_source_extractor(txt, "machineshop")))
"""

# ╔═╡ b54c3620-0fb5-11eb-2348-b9f57946f679
let sim = Simulation()
	sto = Store{Machine}(sim, capacity=UInt(2))
	ms = @process machineshop(sim, sto)
	users = [@process user(sim, i, sto, (i % 2) + 1) for i=0:2]
	run(sim)
end

# ╔═╡ e04cf3fa-e5e2-4817-a3fd-82ccebfdee58
md"""
## `ConcurrentSim` Applications
### Repair problem
This problem is taken from *Ross, Simulation 5th edition, Section 7.7, p. 124-126*


A system needs ``n`` working machines to be operational. To guard against machine breakdown, additional machines are kept available as spares. Whenever a machine breaks down it is immediately replaced by a spare and is itself sent to the repair facility, which consists of a single repairperson who repairs failed machines one at a time. Once a failed machine has been repaired it becomes available as a spare to be used when the need arises. All repair times are independent random variables having the common distribution function ``G``. Each time a machine is put into use the amount of time it functions before breaking down is a random variable, independent of the past, having distribution function ``F``.

The system is said to “crash” when a machine fails and no spares are available. Assuming that there are initially ``n + s`` functional machines of which ``n`` are put in use and ``
s`` are kept as spares, we are interested in simulating this system so as to approximate ``E[T]``, where ``T`` is the time at which the system crashes.

#### Building the simulation's components
"""

# ╔═╡ e566e732-129e-45eb-8b24-98064dc42a75
begin
	# constants for the problem
	const RUNS = 30
	const N = 10
	const S = 3
	const LAMBDA = 100
	const MU = 1

	const F = Exponential(LAMBDA)
	const G = Exponential(MU)
end;

# ╔═╡ 6b1ac314-546c-4a36-b618-ba371ca8a0da
md"""
The behavior of a single machine can be done as follows:
$(Markdown.parse(function_source_extractor(txt, "machine(")))

The startup procedure is done as follows:
$(Markdown.parse(function_source_extractor(txt, "start_sim")))

With these elements, we can build a function that models a single run and gives us the time to failure:
$(Markdown.parse(function_source_extractor(txt, "sim_repair")))
"""

# ╔═╡ 9e67ae32-5cc0-482a-b186-e61a5eeb47bf
# single simulation run
sim_repair(N, S, F, G)

# ╔═╡ b92162ba-b03b-4c50-b23d-a37d2c24964a
md"""
#### Monte Carlo approach

"""

# ╔═╡ 9d268a7e-1dac-4778-95f2-032566c9184b
begin
	Logging.disable_logging(LogLevel(1000));
	repair_results = Vector{Float64}(undef, RUNS)
	Threads.@threads for i=1:RUNS
		repair_results[i] = sim_repair(N, S, F, G)
	end
end

# ╔═╡ b97efb24-fd1f-4fd3-ad17-f444c149511e
md"#### Result visualisation"

# ╔═╡ 0481a4b0-bdbb-488c-80cf-16581b2f2c25
boxplot(repair_results, label="", ylabel="Time to failure", xticks=false, size=(200,400))

# ╔═╡ 33cecba2-41be-44ca-93ce-d13464b0c564
md"""
### Other
More extensive examples will be covered during the practical sessions.
 """

# ╔═╡ Cell order:
# ╟─81adcbe6-e0db-4eca-acc1-9a871ad884ff
# ╟─ef11256b-c5ee-43d4-8757-918a05ad3d1d
# ╠═60521568-6a00-4c6b-b11a-1d53d75417c2
# ╟─1546ff80-0faf-11eb-2e03-a3ed4337a0df
# ╠═d3d96c66-de37-49b6-94a5-be888f45f3a3
# ╟─4d63bf51-df9f-4c7b-8008-5ebf956571fe
# ╟─0153c520-0fb0-11eb-1253-0bd7e9dd04ee
# ╠═2326d3e2-0fb0-11eb-1827-6d08b4032c34
# ╟─31d8458e-0fb0-11eb-0fe5-0b2a1a5d56c3
# ╟─b106b67e-0fb0-11eb-04f2-a734c4fa2e3b
# ╟─eda6bb80-0fb0-11eb-2f87-d586f0b69f26
# ╠═f67c3cce-0fb0-11eb-3349-4181800dde90
# ╟─061cc970-0fb1-11eb-0210-d7d0f215be77
# ╟─32c0db60-0fb1-11eb-3d75-f1a189baf482
# ╟─f8b8f0ee-0fb1-11eb-13d6-571cad22a19a
# ╠═05760cb2-0fb2-11eb-2bc4-dd2a2bfde126
# ╟─1d92d712-0fb2-11eb-23fd-4554371e8a3b
# ╠═7845ae80-0fb2-11eb-195b-4d0d4ffca60b
# ╟─9a77cd30-0fb2-11eb-1196-df634baf199d
# ╠═d683334e-0fb2-11eb-3dc4-4be3043674a7
# ╟─eff30a90-0fb2-11eb-01c6-c3a35033923d
# ╟─066d6f8e-0fb3-11eb-0979-ab193faf5653
# ╠═13ba2120-0fb3-11eb-0599-831b085542c8
# ╟─461ec14e-0fb5-11eb-38f0-61ad30ab4e1e
# ╟─5d04b960-0fb5-11eb-01fe-fd0ff58627f6
# ╠═23885800-0fb4-11eb-0378-1191e7ec5ca8
# ╟─835bf0b0-0fb5-11eb-2a68-275326c4e73c
# ╠═b54c3620-0fb5-11eb-2348-b9f57946f679
# ╟─e04cf3fa-e5e2-4817-a3fd-82ccebfdee58
# ╠═e566e732-129e-45eb-8b24-98064dc42a75
# ╟─6b1ac314-546c-4a36-b618-ba371ca8a0da
# ╠═9e67ae32-5cc0-482a-b186-e61a5eeb47bf
# ╟─b92162ba-b03b-4c50-b23d-a37d2c24964a
# ╠═9d268a7e-1dac-4778-95f2-032566c9184b
# ╟─b97efb24-fd1f-4fd3-ad17-f444c149511e
# ╟─0481a4b0-bdbb-488c-80cf-16581b2f2c25
# ╟─33cecba2-41be-44ca-93ce-d13464b0c564
