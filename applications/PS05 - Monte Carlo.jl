### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ bb35ea5b-7629-4831-9c21-0658d0979cee
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	while !isfile("Project.toml") && !isdir("Project.toml")
        cd("..")
    end
    Pkg.activate(pwd())
end

# ╔═╡ 5493a468-0c7a-4d63-bb0e-1adbf8d408ce
begin
	using Random
	using Plots, StatsPlots, LaTeXStrings, Measures
	using Statistics
	using Distributions
	using Graphs
	using BenchmarkTools
	using CSV, DataFrames
	using ConcurrentSim
	using Logging
end

# ╔═╡ ce6d0508-8283-11f0-2bcb-9b53de934eb4
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

# ╔═╡ 763fcc5e-94c5-4406-922e-3393b2f7b8c3
md"""
# Monte Carlo methods
As you will have discovered during the lectures, the term *Monte Carlo* covers a range of techniques that use repeated sampling to solve a complex problem. It can briefly be summarised as:

!!! tip "Monte Carlo"
	Monte Carlo methods are mathematical techniques that use repeated random sampling to estimate numerical results for problems that are difficult to solve analytically. 

	The core idea is to define a domain of inputs, randomly generate samples from that domain according to a probability distribution, perform computations on these samples, and then aggregate the results to approximate quantities such as integrals, expectations, or probabilities.

	By relying on the law of large numbers, Monte Carlo approximations improve in accuracy as the number of random samples increases. These methods are widely used in numerical integration, optimization, and probabilistic modeling where analytical solutions are infeasible or complex.
"""

# ╔═╡ Cell order:
# ╟─ce6d0508-8283-11f0-2bcb-9b53de934eb4
# ╟─bb35ea5b-7629-4831-9c21-0658d0979cee
# ╠═5493a468-0c7a-4d63-bb0e-1adbf8d408ce
# ╠═763fcc5e-94c5-4406-922e-3393b2f7b8c3
