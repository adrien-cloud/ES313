### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ af43ec68-24a5-11ef-395c-f52d0ef97f09
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

	
	import PlutoUI: TableOfContents, LocalResource
	TableOfContents(depth=4)
end

# ╔═╡ bad420d3-8971-4b03-816d-8be354f5009d
# dependencies
begin
	using ResumableFunctions
	using ConcurrentSim
	using Logging

	using Distributions
	using Plots
	using StatsPlots
	using LaTeXStrings
	using Measures            # For fine-grained plot control 
end

# ╔═╡ 3ffef524-2386-4010-878c-c11c7e0515a8
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

# ╔═╡ 1f9f0785-abc5-406e-a517-32c218904e4c
md"""
# Simulation overview
In this last chapter, we will tackle the specific problem of queueing systems and approach it from different angles. Additionally, we'll discuss important factors to consider when developing and implementing simulations. 

The queuing theory part is inspired by:

*Shortle, J. F., Thompson, J. M., Gross, D., Harris, C. M. (2018). Fundamentals of Queueing Theory. Germany: Wiley.*

## General queuing systems
A general queuing system is depicted on the image below

$(LocalResource("./lectures/img/queueing_system.png", :width=>750))

We can identify the following components:
- "customer" arrivals
- a service facility comprised of:
  - a queue for the "customers"
  - different servers
- "customers" leaving

We can also observe different possible transitions, indicated by the arrows."""

# ╔═╡ ee7affdb-177e-4da9-abe5-fd7f23820ece
md"""

### Descriptors
In general, we can describe a queuing system by a set of metrics:

| Symbol            | Interpretation                                      |
|-------------------|-----------------------------------------------------|
| ``\lambda``       | Average arrival rate                                 |
| ``S``             | Random service time                                  |
| ``\mu \equiv 1/E[S]`` | Average service rate                           |
| ``c``             | Number of servers                                    |
| ``r \equiv \lambda/\mu`` | Offered load                                |
| ``\rho \equiv \lambda/c\mu`` | Traffic intensity or utilization        |
| ``T, T_q``        | Random time a customer spends in the system / queue  |
| ``W, W_q``        | Average time a customer spends in the system / queue |
| ``N, N_q``        | Random number of customers in the system / queue     |
| ``L, L_q``        | Average number of customers in the system / queue    |
| ``A^{(n)}``       | Arrival time customer ``n``         			   |
| ``D^{(n)}``       | Departure time customer ``n``  			           |
| ``P_{n}``       | Probability of ``n`` customers in the system          |






### Interest
Queuing models find applications in various fields, including:

* Telecommunications: Modeling call centers, network traffic, and server systems.
* Manufacturing: Analyzing production lines and inventory systems.
* Transportation: Studying traffic flow and congestion.
* Computer Science: Modeling computer systems and network performance.


Generally there are three types of system responses of interest: 
1. Some measure of the waiting time that a typical customer might endure.
2. Some measure of the number of customers that may accumulate in the queue or system. 
3. A measure of the idle time of the servers. 

Since most queueing systems have stochastic elements, these measures are often random variables, so their probability distributions – or at least their expected values – are sought.

The task of the queuing analyst is generally one of two things:
1. Determine some measures of effectiveness for a given process: here one must determine waiting delays and queue lengths from the given properties of the input stream and the service procedures
2. to design an “optimal” system according to some criterion: here one might want to balance customer-waiting time against the idle time of servers according to some cost structure. If the costs of waiting and idle service can be obtained directly, they can be used to determine the optimum number of servers. To design the waiting facility, it is necessary to have information regarding the possible size of the queue. There may also be a space cost that should be considered along with customer-waiting and idle-server costs to obtain the optimal system design. 

In any case, the analyst can first try to solve this problem by analytical means; if these fail, he or she may use simulation. Ultimately, the issue generally comes down to a trade-off between better customer service and the expense of providing more service capability, that is, determining the increase in investment of service for a corresponding decrease in customer delay.
"""

# ╔═╡ 256512c8-6b13-48fb-a5eb-9aefb241ce3a
md"""
### Little's law
Little's Law is a fundamental theorem in queueing theory that relates the average number of customers in a system to the average arrival rate and the average time a customer spends in the system.

The general form of Little's law is 
```math
\mathbb E\left[L\right]=\lambda\mathbb E\left[W\right]
```

with
* ``\mathbb E\left[L\right]``: expected number of customers in the system (both in the queue and being served).
*  ``\lambda``: average arrival rate of customers to the system.
* ``E\left[W\right]`` :  expected time a customer spends in the system.


There is also a specific form of Little's Law that applies just to the queue, rather than the entire system. This form relates the average number of customers in the queue to the average arrival rate and the average time a customer spends waiting in the queue:

```math
\mathbb E\left[L_Q\right]=\lambda\mathbb E\left[W_Q\right]
```

with
* ``\mathbb E\left[L_{Q}\right]``: expected number of customers in the queue.
*  ``\lambda``: average arrival rate of customers to the system.
* ``E\left[W_{Q}\right]`` : expected time a customer spends in the queue.

"""

# ╔═╡ 078324c1-128e-4805-813f-4da02b967d61
md"""
### Steady state probabilities
In the context of queueing theory, the steady state probabilities refer to the long-term probabilities of being in a particular state (e.g., having a certain number of customers in the system) once the system has reached equilibrium.

```math
P_n=\lim_{t\rightarrow\infty}\mathbb P\left\{L(t)=n\right\}
```

where
* ``P_n`` represents the steady state probability of having ``n`` customers in the system.
* ``\lim_{t\rightarrow\infty}\mathbb P\left\{L(t)=n\right\}`` is the probability that the number of customers ``L(t)`` in the system is ``n`` as time ``t`` approaches infinity.
"""

# ╔═╡ 3e361b9a-2ef9-46f1-8e64-127f0a53af42
md"""
### Stability
In queueing theory, stability is an important concept. A queueing system is stable if the arrival rate does not exceed the service rate over the long run. 
```math
a_n=d_n
```
This equation indicates that for the system to be stable, the arrival rate must equal the departure rate for each state ``n``. This ensures that the number of customers in the system does not grow unbounded over time.

"""

# ╔═╡ 49b288bf-b0a2-4e29-bfc0-779bd80c0bc2
md"""
### Poisson Arrivals See Time Averages (PASTA)

The PASTA property is an important result in queueing theory which states that for a system with Poisson arrivals, the probability of an arrival seeing a particular state of the system is equal to the long-term time-average probability of the system being in that state:
```math
P_n=a_n
```

This equation indicates that for a system with Poisson arrivals, the steady state probability ``P_n``  of having ``n`` customers in the system is equal to the arrival rate ``a_n`` when there are ``n`` customers in the system.
"""

# ╔═╡ 2e5e85ed-2234-48ee-a73d-33d73d06112a
md"""
## The M/M/1 queuing system

The M/M/1 queue is a fundamental model in queueing theory. The name of this model stems from its characteristics:

* M (Markovian arrival process): Customers arrive at the queue according to a Poisson process, meaning arrivals occur randomly and independently at a constant average rate (λ).
* M (Markovian service times): The service time for each customer follows an exponential distribution, meaning service times vary randomly but have a constant average rate (μ).
* 1 (Single server): There is only one server available to serve customers in the queue.

The figure below shows an illustration of such a queue: 

$(LocalResource("./lectures/img/MM1_queue.png"))

There are some assumptions associated with this model:
1. Infinite Capacity: The queue can hold an unlimited number of customers.
2. First-Come, First-Served (FCFS): Customers are served in the order they arrive.
3. Steady State: The system has reached a stable state where the arrival rate and service rate are constant over time.

While the M/M/1 model is a valuable tool, it has some limitations:
* Assumptions: The assumptions of Poisson arrivals and exponential service times may not always hold in real-world scenarios.
* Single Server: It only applies to systems with a single server. More complex models are needed for systems with multiple servers.

### Descriptives
It can be shown that the M/M/M1 queuing model has the following properties:

| Metric                | Value             |
|------------------------------------------------------------------|----------------------------------------|
| ``\rho`` (utilization)                                           | `` \frac{\lambda}{\mu}`` |
| ``P_0`` (steady state probability for zero customers)            | ``1 - \frac{\lambda}{\mu}``                       |
| ``P_n`` (steady state probability for ``n`` customers)           | ``\left(1 - \frac{\lambda}{\mu}\right)\left(\frac{\lambda}{\mu}\right)^n \; (\forall n>0)`` |
| ``\mathbb{E}[L]`` (Expected number of customers in the system)   | ``\frac{\lambda}{\mu - \lambda}`` |
| ``\mathbb{E}[W]`` (Expected number of customers in the system)   | ``\frac{1}{\mu - \lambda}`` |
| ``\mathbb{E}[W_Q]`` (Expected time a customer spends waiting in the queue)   | ``\frac{\lambda}{\mu}\frac{1}{\mu - \lambda}`` |
| ``\mathbb{E}[L_Q]`` (Expected number of customers in the queue)   | ``\frac{\lambda^2}{\mu}\frac{1}{\mu - \lambda}`` |


"""

# ╔═╡ e3f8cfb6-f285-45f0-a551-6412e3b2f35c
md"""
### Simulation
A MM1 queuing system will be used to illustrate the three main simulation methodologies:
- time-stepping
- discrete-events processing
- process-driven simulation
"""

# ╔═╡ 98c7d5e9-0476-48f0-8dfc-21de440018b1
begin
	const λ = 1.0
	const μ = 2.0
end;

# ╔═╡ 7431f6ed-f69a-43ca-9200-0c4304357d7f
md"""
#### Time-stepping
A small value for the time increment ``\Delta t`` is chosen and every tick of the clock a function that mimics our queuing system is run.

Exponential distributions can be easily simulated; ``P(\text{arrival})=\lambda\Delta t`` and ``P(\text{departure})=\mu\Delta t``.
"""

# ╔═╡ d1afab78-ed9f-401b-81ed-3a4131d619d1
Δt = 0.1

# ╔═╡ 1df06a1b-373f-4c47-a0a0-aa076905bafc
function time_step(nr_in_system::Int)
    if nr_in_system > 0
        if rand() < μ*Δt
            nr_in_system -= 1
        end
    end
    if rand() < λ*Δt
        nr_in_system += 1
    end
    nr_in_system
end

# ╔═╡ aa80bf93-fefe-4b80-87fa-a9cdc22b9778
let output = Int[], t = 0.0, tmax=10
	push!(output, 0)
	while t < tmax
		t += Δt
		result = time_step(output[end])
		push!(output, result)
	end
	plot(range(0, tmax+Δt, step=Δt), output, line=:steppost, label="", xlabel="t", ylabel="N", marker=:cross, markeralpha=0.5)
end

# ╔═╡ 2d35b0cd-2820-4e85-b090-bb83e4e082af
md"""
We can make the following observations:
* This approach is very easy to implement for simple queuing systems, but becomes cumbersome if the system gets more complex (e.g. larger number of queues, additional interactions, other distributions, ...)
* When are we in steady-state? 
* How many samples of the system in steady-state are needed, to produce a useful average?
* How many runs do a need to have some statistics about the variation around the average?

"""

# ╔═╡ 86d8fd3f-c588-4205-ab85-c19a131640b4
md"""
#### Discrete-event processing
Looking at the output of the time-stepping procedure, we can observe for a lot of the time-steps the state of our system (i.e. the number of clients in the system) does not change. So the procedure does a lot of processing for nothing.

To be more efficient, we can predict
- the next arrival of a client by sampling an exponential distribution with parameter ``\frac{1}{\lambda}``;
- the service time of a client by sampling an exponential distribution with parameter ``\frac{1}{\mu}``.


Only during an arrival of a client or an end of service of a client, the state of the systems changes.
"""

# ╔═╡ dcee2831-3f71-4ea0-a124-401a4a93452e
begin
	const interarrival_distribution = Exponential(1/λ)
	const service_distribution = Exponential(1/μ)
end;

# ╔═╡ 07520778-2231-46d1-99a8-7db44b7491c4
function service(ev::AbstractEvent, times::Vector{Float64}, output::Vector{Int})
    sim = environment(ev)
    time = now(sim)
    push!(times, time)
    push!(output, output[end]-1)
    if output[end] > 0
        service_delay = rand(service_distribution)
        @callback service(timeout(sim, service_delay), times, output)
    end
end

# ╔═╡ b97beac5-fdf8-484a-89fc-f7421b6e9bd1
function arrival(ev::AbstractEvent, times::Vector{Float64}, output::Vector{Int})
    sim = environment(ev)
    time = now(sim)
    push!(times, time)
    push!(output, output[end]+1)
    if output[end] == 1
        service_delay = rand(service_distribution)
        @callback service(timeout(sim, service_delay), times, output)
    end
    next_arrival_delay = rand(interarrival_distribution)
    @callback arrival(timeout(sim, next_arrival_delay), times, output)
end

# ╔═╡ e4db485f-ef16-4d73-9787-7e89c9b056f7
let times = Float64[0.0], output = Int[0], sim = Simulation(), next_arrival_delay = rand(interarrival_distribution)
	@callback arrival(timeout(sim, next_arrival_delay), times, output)
	run(sim, 10.0)
	plot(times, output, line=:steppost, label="", xlabel="t", ylabel="N", marker=:circle, markeralpha=0.8, markerfill=:lightblue)
end

# ╔═╡ d9d29b25-ab6e-4425-8f90-4901ddd9f763
md"""
We can make the following observations:
- Two callback functions describe completely what happens during the execution of an event.
- For complicated systems (network of queues, clients with priorities, other scheduling methods) working with discrete events in this ways results in spaghetti code.
- Code reuse is very limited. A lot of very different application domains can be modeled in a similar way.
"""

# ╔═╡ d2cb1cbb-cd2b-463d-8a39-376874c0d3d8
md"""#### Process-driven discrete-event simulation
We now come to the final approach, where we used process-driven distrete-event simulation. The advantage of this approach is that events and their callbacks are abstracted and the simulation creator has only to program the logic of the system. A process function describes what a specific entity (also called agent) is doing.

We can use `ConcurrentSim` to build a simulation of our M/M/1 system.

Generation process:
$(Markdown.parse(function_source_extractor(txt, "packet_generator(")))

Life cycle of a packet:
$(Markdown.parse(function_source_extractor(txt, "packet(")))

A single simulation:
$(Markdown.parse(function_source_extractor(txt, "MM1_queue_simulation")))
"""

# ╔═╡ fda9c83a-4c19-4cb2-9dfe-1760f287952d
MM1_queue_simulation(interarrival_distribution, service_distribution, 10)

# ╔═╡ 70143d09-77ec-4bd5-b621-ec638fbfb000
let
	t,n = MM1_queue_simulation(interarrival_distribution, service_distribution, 10)
	plot(t, n, line=:steppost, label="", xlabel="t", ylabel="N", marker=:circle, markeralpha=0.8, markerfill=:lightblue)
end

# ╔═╡ 5500645f-67f0-486d-834b-a6aebbf74880
md"""
#### Monte Carlo approach
We now have an efficient method to obtain data from a single simulation. We can proceed to a Monte Carlo approach for a more robust analysis of the system.
"""

# ╔═╡ 9b3ec442-df47-4dce-8dc0-373d8737252b
begin
	const RUNS = 30
	const DURATION = 1000.0
end;

# ╔═╡ 5b9c83fd-8031-450a-8217-d8c6d1f884a6
md"""
We can try to retrieve the analytical descriptives using simulation

##### Estimating ``P_n``


**Note:** you should normalize the probabilities of states for the time spent in them.
"""

# ╔═╡ bfd5a133-004d-4889-b541-f40e05b51e54
let
	N = range(0, 10)
	probmatrix = zeros(RUNS, maximum(N)+1)

	# First plot - overlap of the simulations
	P1 = plot([], [], label="Simulations", color=:red, alpha=0.2)
	
	for i in 1:RUNS
		Pₙ = Dict{Int, Float64}()
		times, clients, _ = MM1_queue_simulation(interarrival_distribution, service_distribution, DURATION)
		for (i,t) in enumerate(times[1:length(times)-1])
			duration = times[i+1] - t
			Pₙ[clients[i]] = get(Pₙ, clients[i],0) + duration	
		end
		Pₙ_sim = [get(Pₙ,n, 0) for n in N] ./ sum(values(Pₙ))

		# assing probability matrix
		for n in N
			probmatrix[i, n+1] = Pₙ_sim[n+1]
		end

		plot!(P1,N, Pₙ_sim, label="", color=:red, alpha=0.25)
	end
	
	
	Pₙ_th = [(1 - λ / μ) * (λ / μ)^n for n in N]
	plot!(P1,N, Pₙ_th, label="Theoretical",linestyle=:dash)
	plot!(P1,xlabel=L"n", ylabel=L"P_n", xticks=0:maximum(N), xlims=(0,maximum(N)), ylims=(0, 0.6), size=(300, 300), title="Individual simulations", titlefontsize=12)

	# boxplot of the simulations
	P2 = boxplot(N', probmatrix, color=:blue, label="", alpha=0.2)
	boxplot!([],[], color=:blue, label="Distribution", alpha=0.2)
	scatter!(P2, N, Pₙ_th, label="Theoretical",  color=:black, marker=:x)
	plot!(P2,xlabel=L"n", ylabel=L"\hat{P}_n", xticks=0:maximum(N), xlims=(-0.5,maximum(N)+0.5), ylims=(0, 0.6), size=(300, 300), title="Estimator distribution", titlefontsize=12)
	# combined plot
	subplots = plot(P1, P2, size=(800, 400))
	global_title = plot(title="M/M/1 - Simulation vs Theory", grid=false, showaxis=false, ticks=false, bottom_margin = -30Plots.px)
	plot(global_title, subplots, layout=@layout([A{0.01h}; B]), left_margin=15Plots.px, bottom_margin=2mm)
end

# ╔═╡ 8e02afc1-1471-4d03-9a37-2d82526e0e80
md"""
#### Estimating ``\mathbb{E}[W_Q]``
To get an estimate of the mean time spent waiting, we need to retrieve this information from our simulation.
"""

# ╔═╡ af159f28-a476-402f-9f96-123b2cbf5f8f
let
	waitmeans = Float64[]
	for i in 1:RUNS
		_,_,wait = MM1_queue_simulation(interarrival_distribution, service_distribution, DURATION)
		push!(waitmeans, mean(wait))
	end

	boxplot(waitmeans, fill=0.5, label="")
	scatter!([1], [λ / (μ * (μ - λ))], marker=:x, color=:black, label="Theoretical")
	plot!(xticks=false, xlims=(0,2), xlabel="", ylabel=L"\mathbb{E}[W_Q]", ylims=(0.2, 1.0), title="Mean waiting time distribution", size=(400, 400))
	#mean(wait), )
end

# ╔═╡ 1270f652-40b7-4088-b1a2-164decdf7f18
md"""
## Considerations
There are some things you should keep in mind when running simulations:

### Runtime
Runtime refers to the amount of time a simulation takes to execute. It encompasses the entire duration from the start of the simulation until it finishes, including all computations and processes involved in generating the results.

Long runtimes can be impractical, especially for complex models or those requiring numerous iterations to achieve statistical significance. Knowing the runtime helps in planning and allocating computational resources effectively.
It can also be used for comparing the efficiency of different simulation algorithms or models.

**Note**: You can often run simulations in parallel, by dispatching the simulations to seperate threads, this can substantially improve the total computation time.

### Number of Runs
The number of runs refers to the number of times a simulation is executed. Each run typically starts with different initial conditions or random seeds to ensure statistical validity and robustness of the results.

Multiple runs are necessary to obtain reliable and statistically significant results, reducing the impact of random variations. A higher number of runs allows for more accurate estimation of confidence intervals for the simulated metrics and it ensures that the simulation model's behavior is consistent and reproducible across different runs.


### Stationarity & Ergodicity 
A stationary system is one where its statistical properties (e.g., mean, variance) do not change over time. The behavior of the system remains consistent regardless of when you start observing it.

An ergodic system is one where time averages (averages taken over a single, long run) are equivalent to ensemble averages (averages taken across multiple independent runs at a specific time point). In other words, observing a single run for a long enough duration is sufficient to capture the full range of system behavior.

Why does it matter? If a system is both stationary and ergodic, it simplifies the interpretation of simulation results. A single long run can provide a reliable representation of the system's overall behavior. However, many real-world systems exhibit non-stationarity (e.g., due to changing customer arrival rates) or non-ergodicity (e.g., due to warm-up periods). In these cases, careful consideration is needed when designing experiments and analyzing simulation output.

Given the prevalence of non-stationary and non-ergodic systems in real-world scenarios, it is crucial to carefully assess these properties before designing and interpreting simulation experiments. Some things you can consider:
* Use statistical tests to assess stationarity (e.g., Augmented Dickey-Fuller test). If non-stationarity is detected, consider transforming the data or modeling the underlying trends to achieve stationarity (cf. time series analysis).
* Analyze simulation output to identify any initial warm-up periods where the system's behavior stabilizes. Discard data from this period to ensure subsequent analysis reflects steady-state behavior.
* Even if a system appears stationary, conduct multiple independent runs with different random number seeds. This helps assess the variability of results and provides a more comprehensive understanding of the system's behavior.
* For non-stationary systems, consider time-dependent analysis methods or techniques that account for trends and seasonality. For non-ergodic systems, focus on ensemble averages obtained from multiple runs, as time averages may not be representative.
* In complex systems, a combination of stationary and non-stationary analysis techniques might be necessary. For instance, you can model short-term dynamics as stationary while accounting for long-term trends through non-stationary methods.


### Regenerative approach
The regenerative approach is a technique for analyzing simulation output. It involves identifying "regeneration points" within a simulation run. These are time points where the system essentially "restarts" statistically, becoming independent of its past behavior. By dividing a long simulation run into these independent cycles, the regenerative approach allows for the calculation of more accurate confidence intervals for performance measures.

The regenerative approach helps address the issue of autocorrelation in simulation output. In many systems, events occurring close in time are not statistically independent. This autocorrelation can lead to underestimation of confidence intervals when using traditional statistical methods. The regenerative approach, by focusing on independent cycles, provides a more reliable way to estimate the variability of simulation results.
"""

# ╔═╡ Cell order:
# ╟─af43ec68-24a5-11ef-395c-f52d0ef97f09
# ╟─3ffef524-2386-4010-878c-c11c7e0515a8
# ╠═bad420d3-8971-4b03-816d-8be354f5009d
# ╟─1f9f0785-abc5-406e-a517-32c218904e4c
# ╟─ee7affdb-177e-4da9-abe5-fd7f23820ece
# ╟─256512c8-6b13-48fb-a5eb-9aefb241ce3a
# ╟─078324c1-128e-4805-813f-4da02b967d61
# ╟─3e361b9a-2ef9-46f1-8e64-127f0a53af42
# ╟─49b288bf-b0a2-4e29-bfc0-779bd80c0bc2
# ╟─2e5e85ed-2234-48ee-a73d-33d73d06112a
# ╟─e3f8cfb6-f285-45f0-a551-6412e3b2f35c
# ╠═98c7d5e9-0476-48f0-8dfc-21de440018b1
# ╟─7431f6ed-f69a-43ca-9200-0c4304357d7f
# ╠═d1afab78-ed9f-401b-81ed-3a4131d619d1
# ╠═1df06a1b-373f-4c47-a0a0-aa076905bafc
# ╟─aa80bf93-fefe-4b80-87fa-a9cdc22b9778
# ╟─2d35b0cd-2820-4e85-b090-bb83e4e082af
# ╟─86d8fd3f-c588-4205-ab85-c19a131640b4
# ╠═dcee2831-3f71-4ea0-a124-401a4a93452e
# ╠═b97beac5-fdf8-484a-89fc-f7421b6e9bd1
# ╠═07520778-2231-46d1-99a8-7db44b7491c4
# ╟─e4db485f-ef16-4d73-9787-7e89c9b056f7
# ╟─d9d29b25-ab6e-4425-8f90-4901ddd9f763
# ╟─d2cb1cbb-cd2b-463d-8a39-376874c0d3d8
# ╠═fda9c83a-4c19-4cb2-9dfe-1760f287952d
# ╟─70143d09-77ec-4bd5-b621-ec638fbfb000
# ╟─5500645f-67f0-486d-834b-a6aebbf74880
# ╠═9b3ec442-df47-4dce-8dc0-373d8737252b
# ╟─5b9c83fd-8031-450a-8217-d8c6d1f884a6
# ╟─bfd5a133-004d-4889-b541-f40e05b51e54
# ╟─8e02afc1-1471-4d03-9a37-2d82526e0e80
# ╟─af159f28-a476-402f-9f96-123b2cbf5f8f
# ╟─1270f652-40b7-4088-b1a2-164decdf7f18
