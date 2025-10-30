### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ a55d8bc0-af6e-11f0-052d-d343bdd4c949
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
	using ArchGDAL
	PlutoUI.TableOfContents()
end

# ╔═╡ aa71529f-d9cb-482c-95ba-458c362eccb5
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

# ╔═╡ Cell order:
# ╠═a55d8bc0-af6e-11f0-052d-d343bdd4c949
# ╠═aa71529f-d9cb-482c-95ba-458c362eccb5
