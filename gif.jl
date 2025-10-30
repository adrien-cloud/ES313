### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ 42ba87f0-af84-11f0-0956-ebd6fc42d0cd
if isfile("./applications/img/lavaflow.gif") # Pour éviter que ça se relance à chaque fois 
	LF = lavaflow(100)
	matrices_dict = LF.flow
	keys_sorted = sort(collect(keys(matrices_dict))) 
	matrices = [matrices_dict[k] for k in keys_sorted] 
	folder = mktempdir() 
	for (i, M) in enumerate(matrices) 
		heatmap(M, size=(400,400),colorbar=false) 
		savefig(joinpath(folder, "frame_$i.png")) 
	end 
	frames = [load(joinpath(folder, "frame_$i.png")) for i in 1:length(matrices)] 
	gr() 
	save("./applications/img/lavaflow.gif", cat(frames..., dims=3)) 
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.0"
manifest_format = "2.0"
project_hash = "da39a3ee5e6b4b0d3255bfef95601890afd80709"

[deps]
"""

# ╔═╡ Cell order:
# ╠═42ba87f0-af84-11f0-0956-ebd6fc42d0cd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
