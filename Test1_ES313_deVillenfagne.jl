### A Pluto.jl notebook ###
# v0.20.19

using Markdown
using InteractiveUtils

# ╔═╡ a02ef410-a508-11f0-0686-5f0ccebb7a0b
begin
    # Pkg needs to be used to force julia to use the current project.
    using Pkg
    # Keep changing the directory to the parent directory of the current file until the folder with the Project.toml is found.
    while !isfile("Project.toml") && !isdir("Project.toml")
        cd("..")
    end
    # Print the current working directory to confirm the location.
    print(pwd())
    Pkg.activate(pwd())
    Pkg.instantiate()
end

# ╔═╡ 9aed61cb-f4a2-490e-8e46-f4a7b736dfd0
begin
	using PlutoUI
	using ArchGDAL
	using Plots
	using Distributions
	using ImageMagick
	using FileIO
end

# ╔═╡ 5d1eebcc-f71d-4c3a-b622-83f9b9ef9a06
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

# ╔═╡ e76afe0e-a573-49ac-a76b-97c96e04ecba
md"""
# Rainfall in the Alps.

## Step 1:
!!! info "Tasks"
	1. Download the file `alps_geoinformation.tif` to your computer.
	2. Search the file using the button below.
	3. Run the script and verify that it was loaded into memory correctly.
"""

# ╔═╡ 3727b2ae-e4b8-4d1b-84ff-4c948cdf7972
dataset = ArchGDAL.read("./alps_geoinformation.tif")

# ╔═╡ c2b4d82c-aaf1-445e-8b7b-f3f3be082f6d
md"""
Extract the altitude information using the code below.
"""

# ╔═╡ f41a565a-2a6c-40fb-93bc-63eb4efb40a5
# Extract the first band as a matrix
height = ArchGDAL.getband(dataset, 1) |> ArchGDAL.read

# ╔═╡ 4bc5fc6f-3d3e-4724-83fd-fecfa2cf835f
md"""
## Step 2:
Now it is up to you...
"""

# ╔═╡ d27e2107-db01-4395-9734-b7999939167d
heatmap(height,c=:gist_earth,title="Height map")

# ╔═╡ 615165e0-6496-4a20-83e2-a984373ad6d6
heatmap(pdf.(Normal(500,100), 1:size(height,2)).*pdf.(Normal(400,100), 1:size(height,2))',title="Rain density")

# ╔═╡ 28048ef5-9612-469c-9c0f-464e7ed08a47
function evolve_rain!(W, rain,param)
    size_W = size(W)
    rain_new = zeros(size(rain))

    for pos in CartesianIndices(W)

        I, J = Tuple(pos)
        neigh = CartesianIndex{2}[]
        for (di, dj) in [(x,y) for x in -1:1 for y in -1:1 if (x,y) != (0,0)]
            ni, nj = I + di, J + dj
            if 1 ≤ ni ≤ size_W[1] && 1 ≤ nj ≤ size_W[2] && W[ni, nj] ≤ W[pos]
                push!(neigh, CartesianIndex(ni, nj))
            end
        end
        isempty(neigh) && continue

        hpos = W[pos]
        h = W[neigh]
		q=rain[pos]

		drops = hpos .- h
		tot = sum(drops)
		if tot == 0
			share = param.flowrate * q / (length(neigh))
			rain_new[pos] += q * (1-param.flowrate)
			for k in neigh
				rain_new[k] += share
			end
		else
			rain_new[pos] += q * (1-param.flowrate)
			for (k, d) in zip(neigh, drops)
				rain_new[k] += param.flowrate * q * d / tot
			end
		end
    end

	rain_new.*=(1-param.evaporation_rate)
	rain.=rain_new

    return rain
end


# ╔═╡ 757a7a1f-c88f-4595-8ed3-48ec291dd053
function run_sim_rain(W,iter)
	param=(flowrate=0.1,evaporation_rate=0.01)
	rain=zeros(size(height))
	final_rain=zeros(size(height)...,iter)

	pdf1=pdf.(Normal(500,100), 1:size(height,1))
	pdf2=pdf.(Normal(400,100), 1:size(height,2))
	pdf_total=pdf1.*pdf2'
	
	for i in 1:iter
		if i <= 20
			rain[CartesianIndices(W)].+=pdf_total.*rand(Gamma(3,730))
		end
	
		final_rain[:,:,i].=evolve_rain!(W,rain,param)
	end
	return final_rain
end

# ╔═╡ 5bf01d28-b6b4-4138-bda6-fd17d0a514b9
final_rain=run_sim_rain(height,200)

# ╔═╡ 41393c1e-da8e-43f2-aada-a5f82b587e47
plot([mean(final_rain, dims=(1,2))...],title="Evolution of mean water level over time")

# ╔═╡ 565f7fa0-54a5-40d3-a4f3-04365e7ff6da
let #takes 5 min to run, We love the performance of CDN computers, especially during tests
	folder = mktempdir() 
	for i in 1:size(final_rain,3)
		heatmap(final_rain[:,:,i],c=:gist_earth, size=(400,400),title="Rain level [m] : iter $(i)") 
		savefig(joinpath(folder, "frame_$i.png")) 
	end 
	frames = [load(joinpath(folder, "frame_$i.png")) for i in 1:size(final_rain,3)] 
	gr() 
	save("./rain.gif", cat(frames..., dims=3)) 
end

# ╔═╡ 1ef10215-9918-4c6c-8c15-bdc473bb8b00
PlutoUI.LocalResource("./rain.gif")

# ╔═╡ Cell order:
# ╠═a02ef410-a508-11f0-0686-5f0ccebb7a0b
# ╠═9aed61cb-f4a2-490e-8e46-f4a7b736dfd0
# ╟─5d1eebcc-f71d-4c3a-b622-83f9b9ef9a06
# ╟─e76afe0e-a573-49ac-a76b-97c96e04ecba
# ╟─3727b2ae-e4b8-4d1b-84ff-4c948cdf7972
# ╟─c2b4d82c-aaf1-445e-8b7b-f3f3be082f6d
# ╟─f41a565a-2a6c-40fb-93bc-63eb4efb40a5
# ╟─4bc5fc6f-3d3e-4724-83fd-fecfa2cf835f
# ╟─d27e2107-db01-4395-9734-b7999939167d
# ╟─615165e0-6496-4a20-83e2-a984373ad6d6
# ╠═757a7a1f-c88f-4595-8ed3-48ec291dd053
# ╠═28048ef5-9612-469c-9c0f-464e7ed08a47
# ╠═5bf01d28-b6b4-4138-bda6-fd17d0a514b9
# ╠═41393c1e-da8e-43f2-aada-a5f82b587e47
# ╠═565f7fa0-54a5-40d3-a4f3-04365e7ff6da
# ╠═1ef10215-9918-4c6c-8c15-bdc473bb8b00
