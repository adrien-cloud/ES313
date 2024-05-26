### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 1d6260d4-f663-11ea-03da-efe9ed63f9bd
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	cd(joinpath(dirname(@__FILE__),".."))
    Pkg.activate(pwd())
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ f231418d-4bbc-46a1-bb65-5b1d14141b31
# Dependencies
begin
using NativeSVG # SVG plotting library
using Plots    # for random related activities
end

# ╔═╡ a62f8ca0-4680-4a73-8333-8c56b385839f
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

# ╔═╡ e6ff0e98-f662-11ea-03a7-e3d09e6272a6
md"""# Physical Modelling

Port of [Think Complexity chapter 7](http://greenteapress.com/complexity2/html/index.html) by Allen Downey."""

# ╔═╡ 2abf49e0-f663-11ea-25f3-2f9229de732e
md"""## Diffusion

In 1952 Alan Turing published a paper called “The chemical basis of morphogenesis”, which describes the behavior of systems involving two chemicals that diffuse in space and react with each other. He showed that these systems produce a wide range of patterns, depending on the diffusion and reaction rates, and conjectured that systems like this might be an important mechanism in biological growth processes, particularly the development of animal coloration patterns.

Turing’s model is based on differential equations, but it can be implemented using a cellular automaton.

Before we get to Turing’s model, we’ll start with something simpler: a diffusion system with just one chemical. We’ll use a 2-D CA where the state of each cell is a continuous quantity (usually between 0 and 1) that represents the concentration of the chemical.

We’ll model the diffusion process by comparing each cell with the average of its neighbors. If the concentration of the center cell exceeds the neighborhood average, the chemical flows from the center to the neighbors. If the concentration of the center cell is lower, the chemical flows the other way.

We’ll use a diffusion constant, ``r``, that relates the difference in concentration to the rate of flow:"""

# ╔═╡ 40373f12-f663-11ea-256c-abd1458a8e85
"""
	applydiffusion(array::Array{Float64, 2}, r::Float64=0.1)

Apply simple diffusion process on an `array`, given the diffusion coefficient `r`
"""
function applydiffusion(array::Array{Float64, 2}, r::Float64=0.1)
	# input validation
    0. < r < 1 ? nothing : throw(ArgumentError("`r` should be between zero and one."))
	# actual work
	nr_y, nr_x = size(array)
    out = deepcopy(array)
    for y in 2:nr_y-1
        for x in 2:nr_x-1
            c = array[y-1, x] + array[y, x-1] + array[y, x+1] + array[y+1, x] - 4*array[y, x]
            out[y, x] += r*c
        end
    end
	
    return out
end

# ╔═╡ 52f4e280-f663-11ea-38a4-c52a7b06b564
md"""visualisation:"""

# ╔═╡ 61437b46-f663-11ea-12c6-2ff992812dca
"""
	visualizearray(array::Array{Float64, 2}, dim)

Make a graphical representation of the values in an `array`. `dim` is a scaling factor for the illustration. 

The plot uses grayscale varying between white (0) and black (1).
"""
function visualizearray(array::Array{Float64, 2}, dim)
    (nr_y, nr_x) = size(array)
    width = dim * (nr_x - 1)
    height = dim * (nr_y - 1)
    Drawing(width=width, height=height) do
		for (j, y) in enumerate(2:nr_y-1)
			for (i, x) in enumerate(2:nr_x-1)
				gray = 80*(1-array[y, x])+10
				fill = "rgb($gray%,$gray%,$gray%"
				rect(x=i*dim, y=j*dim, width=dim, height=dim, fill=fill)
			end
		end
	end
end

# ╔═╡ 358ac3e2-f666-11ea-2058-11eedeabff5f
"""
	Diffusion

A struct used to hold and instantiate a diffusion process. 
Initialy, everyting is set to zero except for a square of ones in the center.
"""
mutable struct Diffusion
	array :: Array{Float64, 2}
	function Diffusion()
		diffusion = new(zeros(Float64, 11, 11))
		diffusion.array[5:7, 5:7] = ones(Float64, 3, 3)
		diffusion
	end
end

# ╔═╡ 863a3fa4-f664-11ea-2597-8f18f2a862ac
diffusion = Diffusion();

# ╔═╡ 966fc752-f666-11ea-1794-195433f4cce5
visualizearray(diffusion.array, 30)

# ╔═╡ b0148fb4-f664-11ea-2d64-976658b08661
@bind togglediffusion html"<input type=button value='Next'>"

# ╔═╡ b25969c4-f665-11ea-222b-df9786a710d2
if togglediffusion === "Next"
	diffusion.array = applydiffusion(diffusion.array)
	visualizearray(diffusion.array, 30)
end

# ╔═╡ 4c3d12f5-0606-4f7f-96f4-d687e7b59fa4
md"Situation after 1,000 iterations:"

# ╔═╡ 95dc78e8-f667-11ea-1c07-8f5cc11b011b
let
	diffusion2 = Diffusion();
	for _ in 1:1000
    	diffusion2.array = applydiffusion(diffusion2.array)
	end
	visualizearray(diffusion2.array, 30)
end

# ╔═╡ c03905f2-f667-11ea-3484-b111f7c14f60
md"""## Reaction-Diffusion
We now consider a more elaborate situation, where we have two chemicals ``a`` and ``b``. Both are subject to a diffusion process, but also interact with eachother. Additionally, we also add ``a`` to and remove ``b`` from the system.

We can identify the following contributions to the fluctations of the quantities in our system:
* The reaction between ``a`` and ``b``. It is assumed that the reactions consumes ``a`` and produces ``b``, so this contribution will be negative for ``a`` and positive for ``b``:
```math
ab^2
``` 
* Adding ``a`` to the system. We do this in such a way that the feed rate is maximal when the quantity of ``a`` is near zero:
```math
f  (1 - a)
``` 
* Removing ``b`` from the system. We do this in such a way that the removal rate is minimal when ``b`` is close to zero as follows:
```math
(f + k) b
``` 
* diffusion for both (similar to the previous case)



As long as the rate parameters are not too high, the values of A and B usually stay between 0 and 1."""

# ╔═╡ cf7c1a2c-f667-11ea-358f-df431ec27476
"""
	applyreactiondiffusion(a::Array, b::Array, ra::Float64=0.5, rb::Float64=0.25, f::Float64=0.055, k::Float64=0.062)

Apply the reaction diffustion model for a situation with two chemicals, of which the concentrations are stored in `a` and `b` respectively.

# Parameters:
- `ra`: The diffusion rate of ``a`` (analogous to `r` in the previous section).
- `rb`: The diffusion rate of ``b``. In most versions of this model, `rb` is about half of `ra`.
- `f`: The “feed” rate, which controls how quickly A is added to the system.
- `k`: The “kill” rate, which controls how quickly B is removed from the system.

*Note*: `ca` and `ca` are the result of applying a diffusion kernel to ``a`` and ``b``. Multiplying by `ra` and `rb` yields the rate of diffusion into or out of each cell.
"""
function applyreactiondiffusion(
        a::Array{Float64, 2}, 
        b::Array{Float64, 2}, 
        ra::Float64=0.5, rb::Float64=0.25, 
        f::Float64=0.055, k::Float64=0.062)
    nr_y, nr_x = size(a)
    a_out = deepcopy(a)
    b_out = deepcopy(b)
    for y in 2:nr_y-1
        for x in 2:nr_x-1
            reaction = a[y, x] * b[y, x]^2
            ca = 0.25*(a[y-1, x] + a[y, x-1] + a[y, x+1] + a[y+1, x]) - a[y, x]
            cb = 0.25*(b[y-1, x] + b[y, x-1] + b[y, x+1] + b[y+1, x]) - b[y, x]
            a_out[y, x] += ra*ca - reaction + f * (1 - a[y, x])
            b_out[y, x] += rb*cb + reaction - (f+k) * b[y, x]
        end
    end
	
    return a_out, b_out
end

# ╔═╡ 23f98030-f668-11ea-389b-ff911adaadfd
"""
	ReactionDiffusion

A struct used to hold and instantiate a reaction-diffusion process. 
The initial state holds a maximum value of `a`, random values for `b` (between 0 and 0.1, except for the center, where the values are between 0.1 and 0.2).
"""
mutable struct ReactionDiffusion
	a::Array{Float64, 2}
    b::Array{Float64, 2}
	function ReactionDiffusion()
		reactiondiffusion = new(ones(Float64, 258, 258), rand(Float64, 258, 258)*0.1)
		reactiondiffusion.b[129-12:129+12, 129-12:129+12] += ones(Float64, 25, 25)*0.1
		reactiondiffusion
	end
end

# ╔═╡ 3c706f38-f669-11ea-2266-f743535073b5
reactiondiffusion = ReactionDiffusion();

# ╔═╡ 1b551fc2-f669-11ea-0d2f-ad0e089592d8
visualizearray(reactiondiffusion.b, 2)

# ╔═╡ e1988846-f668-11ea-1b26-8f7dca1bb4e7
@bind togglereactiondiffusion html"<input type=button value='Next'>"

# ╔═╡ dda066c0-f668-11ea-3096-bdea16f40db7
# You can consider different values for f and k.
# f = 0.035, 0.055, 0.039 k = 0.057, 0.062, 0.065
begin 
f = 0.039
k = 0.065
if togglereactiondiffusion === "Next"
	for _ in 1:500
		reactiondiffusion.a, reactiondiffusion.b = applyreactiondiffusion(reactiondiffusion.a, reactiondiffusion.b, 0.5, 0.25, f, k)
	end
	visualizearray(reactiondiffusion.b, 2)
end
end

# ╔═╡ 1734b35e-f66a-11ea-10b4-317d12008809
md"""Since 1952, observations and experiments have provided some support for Turing’s conjecture. At this point it seems likely, but not yet proven, that many animal patterns are actually formed by reaction-diffusion processes of some kind."""

# ╔═╡ 40f84b2e-f66a-11ea-1dc7-3ded9e11af06
md"""## Percolation

Percolation is a process in which a fluid flows through a semi-porous material. Examples include oil in rock formations, water in paper, and hydrogen gas in micropores. Percolation models are also used to study systems that are not literally percolation, including epidemics and networks (of electrical resistors).

We’ll explore a 2-D CA that simulates percolation.

- Initially, each cell is either “porous” with probability `q` or “non-porous” with probability `1-q`.
- When the simulation begins, all cells are considered “dry” except the top row, which is “wet”.
- During each time step, if a porous cell has at least one wet neighbor, it becomes wet. Non-porous cells stay dry.
- The simulation runs until it reaches a “fixed point” where no more cells change state.

If there is a path of wet cells from the top to the bottom row, we say that the CA has a “percolating cluster”.

Two questions of interest regarding percolation are the following:
1. What is the probability that a random array contains a percolating cluster?
2. How does that probability depend on `q`?
"""

# ╔═╡ 77dcd828-f66a-11ea-0942-c5d613d828b7
"""
	applypercolation(array::Array{Float64, 2})

Given an `array`, we update using the prescribed percolation rules.
"""
function applypercolation(array::Array{Float64, 2})
    nr_y, nr_x = size(array)
    out = deepcopy(array)
    for y in 2:nr_y-1
        for x in 2:nr_x-1
            if out[y, x] > 0.0
                c = array[y-1, x] + array[y, x-1] + array[y, x+1] + array[y+1, x]
                if c ≥ 0.5
                    out[y, x] = 0.5
                end
            end
        end
    end
	
    return out
end

# ╔═╡ 818e5534-f66a-11ea-07b2-7722084e1e07
md"visualisation:"

# ╔═╡ 44c8b17a-f66b-11ea-39af-f554593e33eb
"""
	Wall

A struct used for the percolation process that holds the array with the values.

*Note*: a border is added for the visualisation and to account for the update rules.
"""
mutable struct Wall
	array::Array{Float64, 2}
	"""
		Wall(n, q)

	Generate a square `Wall` of size `n` x `n` with a probability `q` of being porous. The top row is initiated to be wet.
	"""
	function Wall(n, q)
    	array = zeros(Float64, n+2, n+2)
		array[2, 2:n+1] = ones(Float64, n)*0.5
		array[3:n+1, 2:n+1] = rand(Float64, n-1, n)
		for y in 3:n+1
			for x in 2:n+1
				if array[y, x] < q
					array[y, x] = 0.1
				else
					array[y, x] = 0.0
				end
			end
		end
		new(array)
	end
end

# ╔═╡ d9ccf8f8-f66b-11ea-2c81-0d771c2e900e
wall = Wall(40, 0.62);

# ╔═╡ e492630e-f66b-11ea-3aed-f37a400243be
visualizearray(wall.array, 10)

# ╔═╡ e048231c-f66b-11ea-191d-8517a6c65bc5
@bind togglepercolation html"<input type=button value='Next'>"

# ╔═╡ 11847488-f66c-11ea-1aca-4b1613dbfb8e
if togglepercolation === "Next"
	wall.array = applypercolation(wall.array)
	visualizearray(wall.array, 10)
end

# ╔═╡ 3563b208-f66c-11ea-3c1a-77f799805d06
md"""## Phase Change

Now let’s test whether a random array contains a percolating cluster. We do this by running the percolation process untill it converges (i.e. no more changes).

"""

# ╔═╡ 2b9c8a86-f66c-11ea-0447-1fb7ba83d3a1
"""
	testpercolation(array::Array{Float64, 2}, vis=false)

Run a percolation process until convergence and subsequently count the number of "wet" cells on the bottom row. If any wet cells are found, the function returns true and false otherwise.

`vis` can be used to toggle the visualisation.
"""
function testpercolation(array::Array{Float64, 2}, vis=false)
    numberwet = count(x->x==0.5, array[3:101, 2:101])
    while true
        array = applypercolation(array)
        if count(x->x==0.5, array[101, 2:101]) > 0
			if vis
				return true, visualizearray(array, 8) 
			else
            	return true
			end
        end
        newnumberwet = count(x->x==0.5, array[3:101, 2:101])
        if numberwet == newnumberwet
            if vis
				return false, visualizearray(array, 8)
			else
            	return false
			end
        end
        numberwet = newnumberwet
    end
end

# ╔═╡ c71f0f24-f66c-11ea-0ba7-d765f3998ed9
let
	# instantiate the wall
	wall = Wall(100, 0.6)
	# run the testpercolation function and render the results
	result, drawing = testpercolation(wall.array, true)
	@info "Percolation $(result ? "reached" : "did not reach") the bottom"
	# show the drawing
	drawing
end

# ╔═╡ fd17fd08-9469-4d65-be44-4b91fc806267


# ╔═╡ e51771ec-f66c-11ea-1606-fd159b8d216f
md"""
### Probability of percolating
To get an id estimate the probability of a percolating cluster, we generate many random arrays and test them."""

# ╔═╡ 1254b21a-f66e-11ea-2b36-7985687edd02
"""
	estimateprob(;n=100, q=0.5, iters=100)

Estimate the percolation probability for a `Wall` of size `n`, initiated with a porous probability `q` based on `iters` estimates.
"""
function estimateprob(;n=100, q=0.5, iters=100)
    t = Bool[]
    for _ in 1:iters
        wall = Wall(n, q)
        push!(t, testpercolation(wall.array))
    end
    count(x->x, t) / iters
end

# ╔═╡ dda2cd12-f66c-11ea-3f31-837c9a749b5c
estimateprob(q = 0.50)

# ╔═╡ a92e9fe7-1cf7-4426-8221-c3e4946f4860
md"""
### Impact of `q` on probability of percolating
We can use our ```estimateprob``` function to get an idea to what extent (and if) `q` has any impact on the probability of percolating.
"""

# ╔═╡ 2b6328b1-0384-4937-994b-ae8270824f42
begin
	# Different q values
	q_vals = collect(range(0.5, stop=0.7, length=20))
	# Associated probabilities
	p_prob = map(q -> estimateprob(q=q), q_vals)
end

# ╔═╡ 952f559a-46bc-40ec-9d1f-1841e2240615
plot(q_vals, p_prob, marker=:circle, label="Percolation probability", xlabel="q")

# ╔═╡ 9f5bbe6d-0ace-4b47-b28e-377962c59232
md"""
The rapid change in behavior around `q=0.58` is called a phase change by analogy with phase changes in physical systems, like the way water changes from liquid to solid at its freezing point.

A wide variety of systems display a common set of behaviors and characteristics when they are at or near a critical point. These behaviors are known collectively as critical phenomena.
"""

# ╔═╡ 340521e8-8c97-4c7d-85b9-93a37ee85563
md"""
### Estimating a critical value for `q`
We now know that a phase change is happening, but you might want to find a value ``q_{\text{crit}}``, such that the percolation probability is above a desired treshold ``P_{\text{crit}}``. 

This can be done in different ways. For example, you could use a bisection method (cf. numerical methods from last year). In the following, we will use a random walk to realise this:
* Starting from an initial value `q_0`, we construct a wall and compute its percolation probability `P`.
* Depending on the value of `P`, we increase or decrease the value of `q` by a fixed amount (e.g. 0.004).
* We stop the random walk when 1) no more changes occur 2) we are close enough the ``P_{\text{crit}}``

This optimisation approach is a form of local search you will also discuss next year in the course "Intelligent decision support methods".
"""

# ╔═╡ d9fc2ec3-b3c3-41b3-8510-b599a1e8e059
"""
	findcritical(;n::Int=100, iters::Int=100, P_crit::Float64=0.7, q₀::Float64=0.5, δq::Float64=0.004,tol::Float64=0.02)

Determine the required value of `q` to match the required percolation probability `P_crit`. 

# Arguments
- n::Int=100: size of the wall
- iters::Int=100 number of iterations for a single estimate of `P` (cf. ```estimateprob```)
- P_crit::Float=0.7: the desired percolation probability
- q₀::Float=0.5: the inital value for the random walk
- δq::Float=0.004: the update step for the q-value
- tol::Float64=0.02: tolerance to stop computation
"""
function findcritical(;n::Int=100, iters::Int=100, P_crit::Float64=0.7, q₀::Float64=0.5, δq::Float64=0.004, tol::Float64=0.02)
	# initialise the values
	qs = [q₀]
	ps = Float64[]
	# run the optimisation
	while true
		# compute the associated probability
		p = estimateprob(;n=n, q=qs[end], iters=iters)
		push!(ps, p)
		if abs(p - P_crit) < tol
			break
		elseif p > P_crit
			push!(qs, qs[end] - δq)
		elseif p < P_crit
			push!(qs, qs[end] + δq)
		end
	end

	return qs, ps
	
end

# ╔═╡ 501878b4-f66e-11ea-270f-d7542715acbb
qs, ps = findcritical(q₀=0.58)

# ╔═╡ 44719f1b-bcbd-4cc3-ab05-aa004f926f6c
plot(ps, xlabel="Iteration", ylabel="P_percolation",label="", marker=:circle)

# ╔═╡ 87c7816a-f66e-11ea-02fe-1752fee2c7eb
md"""## Fractals

To understand fractals, we have to start with dimensions.

For simple geometric objects, dimension is defined in terms of scaling behavior. For example, if the side of a square has length `l`, its area is `l^2`. The exponent, 2, indicates that a square is two-dimensional. Similarly, if the side of a cube has length `l`, its volume is `l^3`, which indicates that a cube is three-dimensional.

More generally, we can estimate the dimension of an object by measuring some kind of size (like area or volume) as a function of some kind of linear measure (like the length of a side).

As an example, we can estimate the dimension of a 1-D cellular automaton by measuring its area (total number of “on” cells) as a function of the number of rows."""

# ╔═╡ af69c480-f66e-11ea-3e4a-1db5d58bdd7b
"""
	inttorule1dim(val::UInt8)

Transform an integer into a rule for Wolfram's experiment
"""
function inttorule1dim(val::UInt8)
	# convert value into binary
    digs = BitArray(digits(val, base=2))
	# pad with additional 'zeros'
    for i in length(digs):7
        push!(digs, false)
    end
	
    return digs
end

# ╔═╡ d6cab994-f66e-11ea-1734-dd3ab2d22e92
"""
	applyrule1dim(rule::BitArray{1}, bits::BitArray{1})

Return the new state based on the own states and the left and right neigbor.
"""
function applyrule1dim(rule::BitArray{1}, bits::BitArray{1})
	# get position in rule
    pos = 1 + bits[3] + 2*bits[2] + 4*bits[1]
	# return next value
    return rule[pos]
end

# ╔═╡ daa15a5a-f66e-11ea-1983-51732624404b
"""
	step1dim(x₀::BitArray{1}, rule::BitArray{1}, steps::Int64)

From a starting configuration `x₀`, apply a `rule` for a total of `steps` times.
"""
function step1dim(x₀::BitArray{1}, rule::BitArray{1}, steps::Int64)
    xs = [x₀]
    len = length(x₀)
    for i in 1:steps
        x = copy(x₀)
        for j in 2:len-1
            x[j] = applyrule1dim(rule, xs[end][j-1:j+1])
        end
        push!(xs, x)
    end
    xs
end

# ╔═╡ e047f8b8-f66e-11ea-05af-398e5d22ccf8
"""
	visualize1dim(res, dim)

Helper function to illustrate the evolution of a one-dimensional Wolfram experiment.

`res` contains the subsequent states and `dim` is a scaling factor for the illustration.
"""
function visualize1dim(res, dim)
    width = dim * (length(res[1]) + 1)
    height = dim * (length(res) + 1)
    Drawing(width = width, height = height) do
        for (i, arr) in enumerate(res)
            for (j, val) in enumerate(arr)
                fill = if val "grey" else "lightgrey" end
                rect(x = j*dim, y = i*dim, width = dim, height = dim, fill = fill)
            end
        end
    end
end

# ╔═╡ ea25224a-f66e-11ea-1c90-b30dba111177
begin
	x₀ = falses(65)
	x₀[33] = true
	res = step1dim(x₀, inttorule1dim(UInt8(18)), 31) # 20, 50, 18
	visualize1dim(res, 10)
end

# ╔═╡ 070b57aa-f66f-11ea-1a12-bd1529268d4f
md"""We estimate the dimension of these CAs with the following function, which counts the number of on cells after each time step."""

# ╔═╡ 041baf7c-f66f-11ea-270b-2179bd949a67
"""
	countcells(rule, n=501)

Count the number of "on" cells in the one-dimensional CA at each of the `n` time steps.

The CA has a width =  2x`n` + 1 + 2 (borders) and start with a single "on" value in the center.
"""
function countcells(rule, n=501)
	# initiate the CA
    x₀ = falses(2*n+3)
    x₀[n+2] = true
	# run n steps
    res = step1dim(x₀, inttorule1dim(UInt8(rule)), n)
	# initial count
    cells = [1]
	# additional counts
    for i in 2:n
        push!(cells, cells[end]+sum(line->count(cell->cell, line), res[i]))
    end
	
    return cells
end

# ╔═╡ 73fd7b5e-f66f-11ea-12cd-dd36181cf956
begin
	n = 501;
	#rule = 20 # 20, 50, 18
	fplot = plot(1:n, 1:n, xaxis=:log, yaxis=:log, label="d = 1", xlabel="Iteration", ylabel="Dimension")
	plot!(fplot, 1:n, (1:n).^2, xaxis=:log, yaxis=:log, label="d = 2",legendposition=:topleft)
	for rule in [20; 50; 18]
		plot!(fplot, 1:n, countcells(rule, n), xaxis=:log, yaxis=:log, label="rule $rule")
	end
	fplot
end

# ╔═╡ 81657f1a-f66f-11ea-183b-0ba3f1c3a38e
md"""
* Rule 20 produces 3 cells every 2 time steps, so the total number of cells after $i$ steps is $y = 1.5i$. Taking the log of both sides, we have $\log y = \log 1.5 + \log i$, so on a log-log scale, we expect a line with slope 1. In fact, the estimated slope of the line is 1.01.
* Rule 50 produces i+1 new cells during the ith time step, so the total number of cells after $i$ steps is $y = i^2 + i$. If we ignore the second term and take the log of both sides, we have $\log y \approx 2 \log i$, so as $i$ gets large, we expect to see a line with slope 2. In fact, the estimated slope is 1.97.
* Finally, for Rule 18, the estimated slope is about 1.57, which is clearly not 1, 2, or any other integer. This suggests that the pattern generated by Rule 18 has a “fractional dimension”; that is, it is a fractal.

This way of estimating a fractal dimension is called [box-counting](https://en.wikipedia.org/wiki/Box_counting)."""

# ╔═╡ feefb6e4-f66f-11ea-2998-a30311bee9f4
md"""## Fractals and Percolation

Now let’s get back to percolation models.

To estimate their fractal dimension, we can run CAs with a range of sizes, count the number of wet cells in each percolating cluster, and then see how the cell counts scale as we increase the size of the array."""

# ╔═╡ 21d918ee-f670-11ea-0359-cbb1809bab9b
"""
	percolationwet(array::Array{Float64, 2})

Count the number of "wet" cells in the two-dimensional percolation at each of the `n` time steps.

The percolation update process stops if either the bottom row is wet, or if we reach a steady-state.
"""
function percolationwet(array::Array{Float64, 2})
    numberwet = count(x->x==0.5, array[3:end-1, 2:end-1])
    while true
        array = applypercolation(array)
        if count(x->x==0.5, array[end-1, 2:end-1]) > 0
            break
        end
        newnumberwet = count(x->x==0.5, array[3:end-1, 2:end-1])
        if numberwet == newnumberwet
            break
        end
        numberwet = newnumberwet
    end
	
    return numberwet
end

# ╔═╡ 3206914e-f670-11ea-3072-d162f584fbf0
let	
	sizes = 10:10:200

	pplot = plot(sizes, sizes, xaxis=:log, yaxis=:log, label="d = 1", xlabel="Iteration", ylabel="Dimension")
	plot!(pplot, sizes, (sizes).^2, xaxis=:log, yaxis=:log, label="d = 2")
	for q in [0.4; 0.8; 0.596]
		res = Float64[]
		for size in sizes
			wall = Wall(size, q)
			push!(res, percolationwet(wall.array))
		end
		plot!(pplot, sizes, res, xaxis=:log, yaxis=:log, seriestype=:scatter, label="q = $q")
	end
	
	pplot
end

# ╔═╡ 4da9c3ba-f670-11ea-1627-d7c5e6b36742
md"""The dots show the number of cells in each percolating cluster. The slope of a line fitted to these dots is often near 1.85, which suggests that the percolating cluster is, in fact, fractal when `q` is near the critical value.

* When `q` is larger than the critical value, nearly every porous cell gets filled, so the number of wet cells is close to `q * size^2`, which has dimension 2.
* When `q` is substantially smaller than the critical value, the number of wet cells is proportional to the linear size of the array, so it has dimension 1."""

# ╔═╡ Cell order:
# ╟─1d6260d4-f663-11ea-03da-efe9ed63f9bd
# ╟─a62f8ca0-4680-4a73-8333-8c56b385839f
# ╠═f231418d-4bbc-46a1-bb65-5b1d14141b31
# ╟─e6ff0e98-f662-11ea-03a7-e3d09e6272a6
# ╟─2abf49e0-f663-11ea-25f3-2f9229de732e
# ╠═40373f12-f663-11ea-256c-abd1458a8e85
# ╟─52f4e280-f663-11ea-38a4-c52a7b06b564
# ╠═61437b46-f663-11ea-12c6-2ff992812dca
# ╠═358ac3e2-f666-11ea-2058-11eedeabff5f
# ╠═863a3fa4-f664-11ea-2597-8f18f2a862ac
# ╠═966fc752-f666-11ea-1794-195433f4cce5
# ╟─b0148fb4-f664-11ea-2d64-976658b08661
# ╟─b25969c4-f665-11ea-222b-df9786a710d2
# ╟─4c3d12f5-0606-4f7f-96f4-d687e7b59fa4
# ╟─95dc78e8-f667-11ea-1c07-8f5cc11b011b
# ╟─c03905f2-f667-11ea-3484-b111f7c14f60
# ╠═cf7c1a2c-f667-11ea-358f-df431ec27476
# ╠═23f98030-f668-11ea-389b-ff911adaadfd
# ╠═3c706f38-f669-11ea-2266-f743535073b5
# ╠═1b551fc2-f669-11ea-0d2f-ad0e089592d8
# ╟─e1988846-f668-11ea-1b26-8f7dca1bb4e7
# ╠═dda066c0-f668-11ea-3096-bdea16f40db7
# ╟─1734b35e-f66a-11ea-10b4-317d12008809
# ╟─40f84b2e-f66a-11ea-1dc7-3ded9e11af06
# ╠═77dcd828-f66a-11ea-0942-c5d613d828b7
# ╟─818e5534-f66a-11ea-07b2-7722084e1e07
# ╠═44c8b17a-f66b-11ea-39af-f554593e33eb
# ╠═d9ccf8f8-f66b-11ea-2c81-0d771c2e900e
# ╠═e492630e-f66b-11ea-3aed-f37a400243be
# ╟─e048231c-f66b-11ea-191d-8517a6c65bc5
# ╠═11847488-f66c-11ea-1aca-4b1613dbfb8e
# ╟─3563b208-f66c-11ea-3c1a-77f799805d06
# ╠═2b9c8a86-f66c-11ea-0447-1fb7ba83d3a1
# ╠═c71f0f24-f66c-11ea-0ba7-d765f3998ed9
# ╠═fd17fd08-9469-4d65-be44-4b91fc806267
# ╟─e51771ec-f66c-11ea-1606-fd159b8d216f
# ╠═1254b21a-f66e-11ea-2b36-7985687edd02
# ╠═dda2cd12-f66c-11ea-3f31-837c9a749b5c
# ╟─a92e9fe7-1cf7-4426-8221-c3e4946f4860
# ╠═2b6328b1-0384-4937-994b-ae8270824f42
# ╠═952f559a-46bc-40ec-9d1f-1841e2240615
# ╟─9f5bbe6d-0ace-4b47-b28e-377962c59232
# ╟─340521e8-8c97-4c7d-85b9-93a37ee85563
# ╠═d9fc2ec3-b3c3-41b3-8510-b599a1e8e059
# ╠═501878b4-f66e-11ea-270f-d7542715acbb
# ╠═44719f1b-bcbd-4cc3-ab05-aa004f926f6c
# ╟─87c7816a-f66e-11ea-02fe-1752fee2c7eb
# ╠═af69c480-f66e-11ea-3e4a-1db5d58bdd7b
# ╠═d6cab994-f66e-11ea-1734-dd3ab2d22e92
# ╠═daa15a5a-f66e-11ea-1983-51732624404b
# ╠═e047f8b8-f66e-11ea-05af-398e5d22ccf8
# ╠═ea25224a-f66e-11ea-1c90-b30dba111177
# ╟─070b57aa-f66f-11ea-1a12-bd1529268d4f
# ╠═041baf7c-f66f-11ea-270b-2179bd949a67
# ╠═73fd7b5e-f66f-11ea-12cd-dd36181cf956
# ╟─81657f1a-f66f-11ea-183b-0ba3f1c3a38e
# ╟─feefb6e4-f66f-11ea-2998-a30311bee9f4
# ╠═21d918ee-f670-11ea-0359-cbb1809bab9b
# ╠═3206914e-f670-11ea-3072-d162f584fbf0
# ╟─4da9c3ba-f670-11ea-1627-d7c5e6b36742
