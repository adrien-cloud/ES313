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


# ╔═╡ Cell order:
# ╟─3f86d540-aeb5-11f0-3937-77b7506ab76d
# ╟─723aae3d-f8ab-42d8-9e55-b04ff4645c14
# ╟─35c960c3-25a9-4606-99f3-c68e0898e617
# ╠═af661db5-959a-4f81-acbe-35a8a58deb9f
