### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 4a80e210-aea6-11f0-1b15-db53fb663f73
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

# ╔═╡ dab9a000-5b88-4c3f-a6b9-587c18970df8
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

# ╔═╡ c51ce920-8321-4b53-bf5c-e1796395b821
md"""
# Test 177POL
"""

# ╔═╡ 5d193ba9-6185-446c-856a-1d79e39dd180
function rumor_advance!(state,size,p_spread,p_stiffle)
	new_state=copy(state)
	for pos in CartesianIndices(state)
		I, J = Tuple(pos)
		neigh=[]
        for (di, dj) in ((0, -1), (0, 1), (1, 0), (-1, 0))
            ni, nj = I + di, J + dj
            if 1 ≤ ni ≤ size[1] && 1 ≤ nj ≤ size[2]
				push!(neigh,state[ni,nj])
            end
        end
		
		if state[pos] == 0 && 1 ∈ neigh && rand() < p_spread
			new_state[pos]=1
		elseif state[pos] == 1 && (1 ∈ neigh || 2 ∈ neigh) && rand() < p_stiffle
			new_state[pos]=2
		end
	end
	state.=new_state
end

# ╔═╡ 36e4159b-ed9b-4c01-a134-2664ea9d795a
let
	iter=1000
	size=(8,8)
	p_spread=0.178
	p_stiffle=0.178

	state=zeros(size)
	
	state[2,3]=1
	state[2,2]=2

	for i in 1:iter
		rumor_advance!(state,size,p_spread,p_stiffle)
		if 1 ∉ state
			@show i
			@show count(==(0),state)
			@show count(==(2),state)
			break
		end
	end

	fig1=heatmap(state, c=:gist_earth, clims=(0,2), title="Current state map", size=(400,400))
end

# ╔═╡ Cell order:
# ╠═4a80e210-aea6-11f0-1b15-db53fb663f73
# ╠═dab9a000-5b88-4c3f-a6b9-587c18970df8
# ╠═c51ce920-8321-4b53-bf5c-e1796395b821
# ╠═36e4159b-ed9b-4c01-a134-2664ea9d795a
# ╠═5d193ba9-6185-446c-856a-1d79e39dd180
