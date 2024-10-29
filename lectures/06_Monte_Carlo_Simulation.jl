### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 1c83c71e-0fa9-11eb-28d3-49496955dc7f
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	cd(joinpath(dirname(@__FILE__),".."))
    Pkg.activate(pwd())
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 1df5ee8c-fd00-404a-8636-bfa534f7ed6c
# dependencies
begin
	using CSV 				# for working with CSV files
	using DataFrames 		# for using dataframes (table-like data)
	using Plots 			# for normal plots
	using StatsPlots 		# for statistical plots
	using LaTeXStrings 		# for fancy strings

	using Logging 			# for more logging control
	using ConcurrentSim 	# for DES
	using Distributions 	# for pobability distributions
end

# ╔═╡ cc246156-a4c1-43da-a1d7-f89e71c2dae7
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

# ╔═╡ bd6092b0-0fa7-11eb-0ec2-cdc01bf1360b
md"""# Monte Carlo Methods
## Monte Carlo Origins

There is no single Monte Carlo method. Rather, the term describes a broad approach encompassing many specific techniques. As its name lightheartedly suggests, the defining element is the application of the laws of chance. Physicists had traditionally sought to create elegant equations to describe the outcome of processes involving the interactions of huge numbers of particles. For example, Einstein’s equations for Brownian motion could be used to describe the expected diffusion of a gas cloud over time, without needing to simulate the random progression of its individual molecules. There remained many situations in which tractable equations predicting the behavior of the overall system were elusive even though the factors influencing the progress of an individual particle over time could be described with tolerable accuracy.

One of these situations, of great interest to Los Alamos, was the progress of free neutrons hurtling through a nuclear weapon as it began to explode. As Stanislaw Ulam, a mathematician who joined Los Alamos during the war and later helped to invent the hydrogenbomb, would subsequently note, “Most of the physics at Los Alamos could be reduced to the study of assemblies of particles interacting with each other, hitting each other, scattering, sometimes giving rise to new particles.”

Given the speed, direction, and position of a neutron and some physical constants, physicists could fairly easily compute the probability that it would, during the next tiny fraction of a second, crash into the nucleus of an unstable atom with sufficient force to break it up and release more neutrons in a process known as fission. One could also estimate the likelihood that neutron would fly out of the weapon entirely, change direction after a collision, or get stuck. But even in the very short time span of a nuclear explosion, these simple actions could be combined in an almost infinite number of sequences, defying even the brilliant physicists and mathematicians gathered at Los Alamos to simplify the proliferating chains of probabilities sufficiently to reach a traditional analytical solution.

The arrival of electronic computers offered an alternative: simulate the progress overtime of a series of virtual neutrons representing members of the population released by the bomb’s neutron initiator when a conventional explosive compressed its core to form a critical mass and trigger its detonation. Following these neutrons through thousands of random events would settle the question statistically, yielding a set of neutron histories that closely approximated the actual distribution implied by the parameters chosen. If the number of fissions increased over time, then a self-sustaining chain reaction was underway. The chain reaction would end after an instant as the core blew itself to pieces, so the rapid proliferation of free neutrons, measured by a parameter the weapon designers called “alpha,” was crucial to the bomb’s effectiveness in converting enriched uranium into destructive power."""

# ╔═╡ 0911b590-0fa8-11eb-0b36-315109cc9657
md"""## Physics of the Atomic Bomb

### Neutron Reaction Rate Proportional to Neutron Flux and Target Area

Assume foil density ``n`` (atoms/cm3), width ``\Delta x``, bombarded with beam (area ``A``) of neutrons ``I`` (neutrons/s) with velocity ``v_n``.

Each nucleus in foil represents possible target area: ``\sigma = \pi R_0^2`` where ``R_0`` is nuclear radius. Total target area ~ ``A \Delta x n \sigma``

Rate of removing neutrons from ``I`` is proportional to: #neutrons crossing through ``A`` and total area presented by all targets:
```math
\frac{\mathrm d N}{\mathrm d t} = \frac{I}{A}\left(A \Delta x n \sigma\right)
```


### Neutron reaction cross sections

Total microscopic neutron cross section is expressed as:
```math
\sigma = \frac{\mathrm d N}{\mathrm d t} \frac{1}{\frac{I}{A}\left(A \Delta x n \right)}
```

Defining neutron flux as: 
```math
\phi= \frac{I}{A} \textrm{(neutrons/s $cm^2$)}
```

Then
```math
\frac{\mathrm d N}{\mathrm d t} = \phi A \Delta x n \sigma
```

Neutron flux can also be defined as ``\phi= n_nv_n`` where ``n_n`` is neutron density per cm3 in beam, ``v_n`` relative velocity (cm/s) of neutrons in beam.

Cross section ``\sigma`` can be experimentally measured as function of energy: ``\sigma\left(E\right)``, expressed in “barns” (b) with 1b = 10e-24cm$^2$.

### Neutron reaction cross sections

Cross sections ``\sigma\left(E\right)`` can be separated into different types of reactions – scattering, absorption, fission:
```math
\sigma\left(E\right) =\sigma_s\left(E\right)+ \sigma_a\left(E\right)+ \sigma_f\left(E\right)
```

Neutron cross section data is available from [NNDC](http://www.nndc.bnl.gov/sigma/index.jsp).
"""

# ╔═╡ 40c34200-0fa9-11eb-12b5-6196f71e1a7c
let
	data = CSV.read("lectures/data/sigma_total.txt", DataFrame)
	plot(data[:,1], data[:,2], xaxis=:log, yaxis=:log, xlabel="E [eV]", ylabel="sigma [b]", label=L"σ_{total}")
	data = CSV.read("lectures/data/sigma_fission.txt", DataFrame)
	plot!(data[:,1], data[:,2], xaxis=:log, yaxis=:log, label=L"σ_{fission}")
	data = CSV.read("lectures/data/sigma_elastic.txt", DataFrame)
	plot!(data[:,1], data[:,2], xaxis=:log, yaxis=:log, label=L"σ_{elastic}")
	data = CSV.read("lectures/data/sigma_inelastic.txt", DataFrame)
	plot!(data[:,1], data[:,2], xaxis=:log, yaxis=:log, label=L"σ_{inelastic}")
	data = CSV.read("lectures/data/sigma_absorption.txt", DataFrame)
	plot!(data[:,1], data[:,2], xaxis=:log, yaxis=:log, label=L"σ_{absorption}")
	plot!(legendfontsize=12, legendposition=:bottomleft)
end

# ╔═╡ 8553233e-0fa9-11eb-119d-01b79be544b7
md"""### Attenuation of Neutron Beam
From conservation of neutrons in beam: number scattered, absorbed, reacted removed from beam: ``\mathrm d N = - \mathrm d I``

Since
```math
\frac{N}{I} = n\Delta x\sigma \leftarrow \begin{cases}
N= In\sigma \Delta x \\
- \mathrm d I = In\sigma \mathrm d x
\end{cases}
```

Integrated, this yields attenuation formula in terms of total reaction cross section and foil density:
```math
I\left(x\right) = I_0\mathrm e^{-n\sigma x}
```

``\frac{I\left(x\right)}{I_0} = \mathrm e^{-n\sigma x}`` is probability of non-interaction

###  Macroscopic Cross Section

For nuclear engineering calculations macroscopic neutron cross section ``\Sigma\left(E\right)= n\sigma\left(E\right)`` becomes more useful

``\Sigma\left(E\right)`` effectively has units of: #/cm3 x cm2 = #/cm

###  Probability of Interaction

Probability of neutron interaction event in ``\mathrm d x`` is expressed as
```math
p\left(x\right) \mathrm d x = \Sigma \mathrm e^{- \Sigma x} \mathrm d x
```

Average distance traveled without interaction, or mean free path:
```math
\lambda = \int_0^{+\infty}xp\left(x\right) \mathrm d x = \frac{1}{\Sigma}
```

Distance traveled without interaction follows an exponential law with parameter ``\Sigma``

### Fission
"""

# ╔═╡ 95544670-0fa9-11eb-36a5-6987b08129a0
begin
	data = CSV.read("lectures/data/sigma_fission.txt", DataFrame)
	const Nₐ = 6.02214086e23 # atoms / mole
	const ρᵤ = 19.1          # g / cm3
	const mᵤ = 235.0439299   # g / mole
	const nᵤ = ρᵤ * Nₐ / mᵤ
	const k = 1.38064852e-23
	const q = 1.60217662e-19
	E = 300 * k / q # eV
	@show E
	i = findfirst(x -> x > E, data[:, 1])
	σ300K = data[i, 2] + (E - data[i, 1]) / (data[i-1, 1] - data[i, 1]) * (data[i-1, 2] - data[i, 2])
	E = 2e6 # eV
	i = findfirst(x -> x > E, data[:, 1])
	σ2e6eV = data[i, 2] + (E - data[i, 1]) / (data[i-1, 1] - data[i, 1]) * (data[i-1, 2] - data[i, 2])
	@show σ300K σ2e6eV # barn
	Σ300K = nᵤ * σ300K * 1e-24
	Σ2e6eV = nᵤ * σ2e6eV * 1e-24
	@show Σ300K Σ2e6eV # cm-1
	λ300K = 1 / Σ300K
	λ2e6eV = 1 / Σ2e6eV
	@show λ300K λ2e6eV; # cm
	σ300K, σ2e6eV, Σ300K, Σ2e6eV, λ300K, λ2e6eV
end

# ╔═╡ 0a643b50-0faa-11eb-2a07-93790802d704
md"""Fission of U235 yields on average: 2.44 total neutrons (1, 2, 3 or 4 depending on reaction)

Neutrons are ejected isotropically.

So due to the spherical symmetry, the angle ``\theta`` with the radius is determined by

```math
\cos\theta \approx \mathcal U\left(\left[-1,1\right]\right)
```

The distance from the center of a neutron created at radius ``r``, flying in the direction ``\theta`` for a distance ``d`` (exponentially distributed) is given by

```math
r^\prime = \sqrt{r^2 + d^2 + 2rd\cos\theta}
```

and the time of flight

```math
\Delta t = \frac{d}{v} = \displaystyle\frac{d}{\sqrt\frac{2E}{m}}
```"""

# ╔═╡ 282bd3f0-0faa-11eb-2357-278dee39e9a0
begin
	v300K = sqrt(2 * 300 * k / 1.674929e-27) # m/s
	Δt300K = λ300K / v300K / 100
	v2e6eV = sqrt(2 * 2e6 * q / 1.674929e-27) # m/s
	Δt2e6eV = λ2e6eV / v2e6eV / 100
	@show v300K v2e6eV Δt300K Δt2e6eV;
	v300K, v2e6eV, Δt300K, Δt2e6eV
end

# ╔═╡ 408cfa50-0faa-11eb-005a-21e46e3bdec4
md"""Energy spectrum of released neutrons is also available from [NNDC](http://www.nndc.bnl.gov/sigma/index.jsp) but we will use the empirical Watt distribution:

```math
P\left(E\right)=0.4865\sinh\left(\sqrt{2E}\right)\mathrm e^{-E}
```"""

# ╔═╡ 649b7a20-0faa-11eb-0ede-a7a1e709af7a
let 
	logE = -8:0.1:1.5
	E = 10 .^(logE)
	plot(E, 0.4865 .* sinh.(sqrt.(2 .* E)) .* exp.(-E), label="Watt distribution", xlabel="E [MeV]", ylabel="Prob")
end

# ╔═╡ 8031178e-0faa-11eb-3ef7-c94a993bc778
md"""1 eV = 1.60217662 10-19 J

Neutrons created by fission are fast neutrons. Scattering is important to increase reaction rate!

### Scattering

Scattering in the center of mass frame:
```math
E_\textrm{out} = E_\textrm{in} \frac{1}{2}\left((1+\alpha) + (1-\alpha)\cos\phi \right)
```

where ``\displaystyle\alpha = \left(\frac{A-1}{A+1}\right)^2`` and A=235 for U235.

The scattering angle in the laboratory frame yields:
```math
\cos\psi = \frac{A\cos\phi + 1}{\sqrt{A^2+2A\cos\phi+1}}
```

The probability of a neutron (initial kinetic energy ``E_\textrm{in}``) colliding and resulting in final neutron kinetic energy ``E_\textrm{out}`` is
```math
P\left\{E_\textrm{in}\rightarrow E_\textrm{out}\right\}=\frac{4\pi\displaystyle\frac{\mathrm d \sigma_s\left(\phi\right)}{\mathrm d \phi}}{\sigma_s E_\textrm{in}\left(1-\alpha\right)}
```

The differential cross section can also is also available from [NNDC](http://www.nndc.bnl.gov/sigma/index.jsp), but we will suppose the scattering happens isotropically in a solid angle so ``\cos\phi`` is distributed uniformally in the interval ``\left[-1,1\right]`` and we use the previous formulas to calculate ``\psi`` and ``E_\textrm{out}``.

The new ``\theta^\prime`` is uniformally distributed in the interval ``\left[\theta-\psi, \theta+\psi\right]``.

### Neutron Multiplication Factor

A numerical measure of a critical mass is dependent on the effective neutron multiplication factor ``k``, the average number of neutrons released per fission event that go on to cause another fission event rather than being absorbed or leaving the material. When ``k=1``, the mass is critical, and the chain reaction is self-sustaining. So for each neutron we should log the amount of neutrons it generates before it dies. Afterwards we can take the average value of all of these and get an idea of the multiplication factor ``k``.

### Spontaneous Fission

U235 has a halflife of 7.037 10^8 years and generates 1.86 neutrons. Spontaneous fission occurs 0.0003 times per g per s."""

# ╔═╡ 22142390-d2e8-4551-8b45-70304c94df64
const numberofspontaneousfis = 0.0003 # / g / s

# ╔═╡ b2955482-0faa-11eb-2a3f-d71bf8a4bc7d
ρᵤ * 4/3 * π * 9^3 * 0.0003

# ╔═╡ b2208e01-7e3b-4d2f-896b-5ad785594cf5
md"""

## Atomic bomb simulation


"""

# ╔═╡ 105dd4bb-5b3a-41f8-aa5a-3f93770779ab
md"### Additional constants"

# ╔═╡ 9b63ecbc-a432-4f68-84f9-4ab7f7b8b004
begin
	const mₙ = 1.008664916    # g / mole
	const Mₙ = mₙ / Nₐ * 1e-3 # kg
	const A = mᵤ / mₙ
	const α = (A - 1)^2 / (A + 1) ^2
	nothing
end

# ╔═╡ 3a6890fe-c1eb-4f16-a09e-d7216b2d08cf
md"### Distributions"

# ╔═╡ db162eda-069d-4176-89d6-69e158715377
begin
	const cosΘdistr = Uniform(-1, 1)
	const cosϕdistr = Uniform(-1, 1)

	const energy = 1e-3:1e-3:15
	function wattspectrum(energy) # MeV
		0.453 * sinh(sqrt(2.29*energy))*exp(-1.036*energy)
	end
	
	const spectrum = wattspectrum.(energy)
	const wattdistr = Categorical(spectrum ./ sum(spectrum))

	const numberofneutronsdistr = Categorical([0,0.6,0.36,0.04])
	const numberofneutronsspontaneousdistr = Categorical([0.2,0.74,0.06])
end;

# ╔═╡ 31d8028f-8455-4acf-920a-b439bc358640
md"### Data "

# ╔═╡ 03557df4-a417-4ae2-ab07-4428c10a9549
begin
	σt = CSV.read("lectures/data/sigma_total.txt", DataFrame)
	σf = CSV.read("lectures/data/sigma_fission.txt", DataFrame)
	σa = CSV.read("lectures/data/sigma_absorption.txt", DataFrame)
	σi = CSV.read("lectures/data/sigma_inelastic.txt", DataFrame)
end;

# ╔═╡ 870c18e4-3ea5-4789-8f7e-e5774bdf8f01
function Σ(energy::Float64) # 1 / cm
    i = findfirst(e -> e > energy, σt[:, 1])
    σ = σt[i, 2] + (energy - σt[i, 1]) / (σt[i-1, 1] - σt[i, 1]) * (σt[i-1, 2] - σt[i, 2])
    nᵤ * σ * 1e-24
end;

# ╔═╡ 8f5acfef-ec3a-403d-a857-01c126c679d9
function ΔtΔl(energy::Float64)
    Δl = -log(rand()) / Σ(energy)
    v = sqrt(2 * energy * q / Mₙ) * 100
    Δl / v, Δl
end;

# ╔═╡ 404f179d-bd8f-4091-bf34-4826a7c19b6b
md"""### Types and Callbacks"""

# ╔═╡ 2ddf00de-3480-42b3-9de1-a36eb1793d2a
struct Bomb
    radius :: Float64             # cm
    generated :: Vector{Int64}
    neutrons :: Vector{Int64}
    times :: Vector{Float64}      # s
    function Bomb(radius::Real)
        new(radius, Float64[], Int64[], Float64[])
    end
end;

# ╔═╡ 69ae0359-3759-45eb-8618-90a6d588d235
begin
	mutable struct Neutron
		r :: Float64                  # cm
		cosθ :: Float64
		energy :: Float64             # eV
		function Neutron(r::Float64, energy::Float64, cosθ::Float64 = rand(cosΘdistr))
			new(r, cosθ, energy)
		end
	end
	
	function Neutron(sim::Simulation, bomb::Bomb, r::Float64, energy::Float64=energy[rand(wattdistr)] * 1e6)
		neutron = Neutron(r, energy)
		time = now(sim)
		@info("$time: create neutron at position $r with cosθ = $(neutron.cosθ) and energy = $(neutron.energy) eV")
		push!(bomb.times, time)
		push!(bomb.neutrons, 1)
		Δt, Δl = ΔtΔl(neutron.energy)
		@callback collision(timeout(sim, Δt), bomb, neutron, Δl)
	end
	
	function collision(ev::AbstractEvent, bomb::Bomb, neutron::Neutron, Δl::Float64)
		sim = environment(ev)
		time = now(ev)
		r′ = sqrt(neutron.r^2 + Δl^2 + 2*neutron.r*Δl*neutron.cosθ)
		if r′ > bomb.radius
			@info("$(now(sim)): neutron has left the bomb")
			push!(bomb.times, time)
			push!(bomb.neutrons, -1)
			push!(bomb.generated, 0)
		else
			i = findfirst(e -> e > neutron.energy, σt[:, 1])
			σtot = σt[i, 2] + (neutron.energy - σt[i, 1]) / (σt[i-1, 1] - σt[i, 1]) * (σt[i-1, 2] - σt[i, 2])
			i = findfirst(e -> e > neutron.energy, σf[:, 1])
			σfis = σf[i, 2] + (neutron.energy - σf[i, 1]) / (σf[i-1, 1] - σf[i, 1]) * (σf[i-1, 2] - σf[i, 2])
			i = findfirst(e -> e > neutron.energy, σa[:, 1])
			σabs = σa[i, 2] + (neutron.energy - σa[i, 1]) / (σa[i-1, 1] - σa[i, 1]) * (σa[i-1, 2] - σa[i, 2])
			i = findfirst(e -> e > neutron.energy, σi[:, 1])
			i = i == 1 ? 2 : i
			σin = σi[i, 2] + (neutron.energy - σi[i, 1]) / (σi[i-1, 1] - σi[i, 1]) * (σi[i-1, 2] - σi[i, 2])
			rnd = rand()
			if rnd < σfis / σtot
				n = rand(numberofneutronsdistr)
				@info("$(now(sim)): fission with creation of $n neutrons")
				for _ in 1:n
					Neutron(sim, bomb, r′)
				end
				push!(bomb.times, time)
				push!(bomb.neutrons, -1)
				push!(bomb.generated, n)
			elseif rnd < (σabs + σfis) / σtot
				@info("$(now(sim)): neutron absorbed")
				push!(bomb.times, time)
				push!(bomb.neutrons, -1)
				push!(bomb.generated, 0)
			elseif rnd < (σin + σabs + σfis) / σtot
				@info("$(now(sim)): inelastic scattering")
				n = 1
				Neutron(sim, bomb, r′)
				push!(bomb.times, time)
				push!(bomb.neutrons, -1)
			else
				cosϕ = rand(cosϕdistr)
				cosψ = (A * cosϕ + 1) / sqrt(A^2 + 2 * A * cosϕ +1)
				neutron.r = r′
				neutron.energy *= 0.5 * (1 + α + (1 - α) * cosϕ)
				θ = acos(neutron.cosθ)
				ψ = acos(cosψ)
				θplusψ = θ + ψ
				θminψ = ψ < π / 2 ? θ - ψ : θ - ψ + 2π
				neutron.cosθ = cos(θplusψ + rand() * (θminψ - θplusψ))
				@info("$(now(sim)): elastic scattering at position $r′ with cosθ = $(neutron.cosθ) and energy = $(neutron.energy) eV")
				Δt, Δl = ΔtΔl(neutron.energy)
				@callback collision(timeout(sim, Δt), bomb, neutron, Δl)
			end
		end
		((sum(bomb.generated) > 500 && sum(bomb.neutrons) == 0) || (time > 1 && sum(bomb.neutrons) == 0) || sum(bomb.generated) > 1000) && throw(StopSimulation())
	end
end;

# ╔═╡ 0a0fcdb3-2818-40df-9747-6100678434ff
function spontaneousfission(ev::AbstractEvent, bomb::Bomb)
    sim = environment(ev)
    for _ in rand(numberofneutronsspontaneousdistr)
        Neutron(sim, bomb, rand() * bomb.radius)
    end
    rate = ρᵤ * 4/3 * π * bomb.radius^3 * numberofspontaneousfis
    @callback spontaneousfission(timeout(sim, -log(rand()) / rate), bomb)
end;

# ╔═╡ be120836-3d00-42c2-a7ee-4a235c6e0486
md"""### Simulation
We can now study the number of neutrons generated for a specific radius
"""

# ╔═╡ 4de0404b-f956-4638-a97c-9a6e77f57f2a
bomb = let
	Logging.disable_logging(LogLevel(-1000));
	myradius = 9
	sim = Simulation()
	bomb = Bomb(myradius)
	@callback spontaneousfission(timeout(sim, 0.0), bomb)
	run(sim)
	bomb
end;

# ╔═╡ f714bc83-603a-4f2e-ac8d-a7d2a9b3542b
@info "Mean neutrons generated: $(mean(bomb.generated))"

# ╔═╡ b7a3c0ce-9e11-4c61-aeba-aed8173db2f1
md"### Visualisation"

# ╔═╡ 072835c4-e84e-48bd-bbff-fec640f3620d
let
	i = findlast(x->x==0, cumsum(bomb.neutrons))
	i = i === nothing ? 1 : i
	plot(bomb.times[i+1:end], cumsum(bomb.neutrons)[i+1:end], seriestype=:scatter, ylabel="N", xlabel="time [s]")
end

# ╔═╡ 74e33241-5ab2-4ddb-a9fd-9009813f508b
md"""
### Monte Carlo approach
Now that we can run a single simulation, we can use this to determine the critical radius.
"""

# ╔═╡ 3ccb69a1-2edc-4f92-b33a-6e5842d185d4
begin
	const RUNS = 100
	const RADII = 5:12;
	Logging.disable_logging(LogLevel(1000));
end;

# ╔═╡ 6466df6e-68dd-4d79-9ba9-027facb64f1b
ks = let
	ks = zeros(Float64, RUNS, length(RADII))
	for (i, r) in enumerate(RADII)
		Threads.@threads for j in 1:RUNS # use multithreading if available  (each run is independent)
			sim = Simulation()
			bomb = Bomb(r)
			@callback spontaneousfission(timeout(sim, 0.0), bomb)
			run(sim)
			ks[j, i] = mean(bomb.generated)
		end
	end
	ks
end;

# ╔═╡ e5faebdc-d97e-409f-920b-35b0a3c9d492
md"The figure below illustrates the distributions and the mean neutron reproduction number"

# ╔═╡ ab915c61-cfca-4635-87a7-f85c1deca3df
begin
	# data plot
	boxplot(reshape(collect(RADII), 1, length(RADII)), ks, alpha=0.25, label="", color=:lightblue)
	# for extra legend entry
	boxplot!([],[], alpha=0.25, label="Distribution")
	# mean neutrons generated
	scatter!(RADII, [mean(ks, dims=1) ...], color=:black, label=L"\langle k \rangle")
	# visuals
	plot!(xlabel="R [cm]", ylabel=L"k", legend=:bottomright, xticks=collect(RADII), title="Evolution ($(RUNS) simulations per radius)")
end

# ╔═╡ Cell order:
# ╟─1c83c71e-0fa9-11eb-28d3-49496955dc7f
# ╟─cc246156-a4c1-43da-a1d7-f89e71c2dae7
# ╠═1df5ee8c-fd00-404a-8636-bfa534f7ed6c
# ╟─bd6092b0-0fa7-11eb-0ec2-cdc01bf1360b
# ╟─0911b590-0fa8-11eb-0b36-315109cc9657
# ╟─40c34200-0fa9-11eb-12b5-6196f71e1a7c
# ╟─8553233e-0fa9-11eb-119d-01b79be544b7
# ╠═95544670-0fa9-11eb-36a5-6987b08129a0
# ╟─0a643b50-0faa-11eb-2a07-93790802d704
# ╠═282bd3f0-0faa-11eb-2357-278dee39e9a0
# ╟─408cfa50-0faa-11eb-005a-21e46e3bdec4
# ╠═649b7a20-0faa-11eb-0ede-a7a1e709af7a
# ╟─8031178e-0faa-11eb-3ef7-c94a993bc778
# ╠═22142390-d2e8-4551-8b45-70304c94df64
# ╠═b2955482-0faa-11eb-2a3f-d71bf8a4bc7d
# ╟─b2208e01-7e3b-4d2f-896b-5ad785594cf5
# ╟─105dd4bb-5b3a-41f8-aa5a-3f93770779ab
# ╠═9b63ecbc-a432-4f68-84f9-4ab7f7b8b004
# ╟─3a6890fe-c1eb-4f16-a09e-d7216b2d08cf
# ╠═db162eda-069d-4176-89d6-69e158715377
# ╟─31d8028f-8455-4acf-920a-b439bc358640
# ╠═03557df4-a417-4ae2-ab07-4428c10a9549
# ╠═870c18e4-3ea5-4789-8f7e-e5774bdf8f01
# ╠═8f5acfef-ec3a-403d-a857-01c126c679d9
# ╟─404f179d-bd8f-4091-bf34-4826a7c19b6b
# ╠═2ddf00de-3480-42b3-9de1-a36eb1793d2a
# ╠═69ae0359-3759-45eb-8618-90a6d588d235
# ╠═0a0fcdb3-2818-40df-9747-6100678434ff
# ╟─be120836-3d00-42c2-a7ee-4a235c6e0486
# ╠═4de0404b-f956-4638-a97c-9a6e77f57f2a
# ╠═f714bc83-603a-4f2e-ac8d-a7d2a9b3542b
# ╟─b7a3c0ce-9e11-4c61-aeba-aed8173db2f1
# ╠═072835c4-e84e-48bd-bbff-fec640f3620d
# ╟─74e33241-5ab2-4ddb-a9fd-9009813f508b
# ╠═3ccb69a1-2edc-4f92-b33a-6e5842d185d4
# ╠═6466df6e-68dd-4d79-9ba9-027facb64f1b
# ╟─e5faebdc-d97e-409f-920b-35b0a3c9d492
# ╟─ab915c61-cfca-4635-87a7-f85c1deca3df
