### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ dab1dfbe-5e52-11f0-05a0-df522fb51dd7
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	while !isfile("Project.toml") && !isdir("Project.toml")
        cd("..")
    end
    Pkg.activate(pwd())
	using PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ cdecef07-e0f9-4372-b1c0-d284d763bb83
# Dependencies
begin
	using NativeSVG # SVG plotting library
	using Plots    # for random related activities
	using Printf   # for fancy text rendering
	using FileIO
	using ImageMagick
	using Graphs
	using SimpleWeightedGraphs
	using GraphPlot
	using Random
	using LinearAlgebra
	using Statistics
	using Distributions
end

# ╔═╡ be2db4dd-9ace-4985-8414-65124e48e7d3
md"""
# Image Segmentation with Cellular Automata

In the lecture notes on physical modelling and self-organisation, a variety of subjects have been extensively covered to introduce you to the possibilities of cellular automata.

In this application, we will apply a CA to a very specific task in image segmentation.

Authors from the University of Guadalajara wrote an article on this application, which can be found at the following link: [Image Segmentation with Cellular Automata](https://www.sciencedirect.com/science/article/pii/S2405844024071834)

Image segmentation is a computer vision technique aimed at dividing an image into distinct and meaningful regions based on similar visual characteristics. The process of image segmentation is often hindered by noise and other undesirable artifacts.

The authors propose a new segmentation approach based on the CA model.

!!! tip "CA segmentation approach"
    The approach consists of 3 phases:
    1. Eliminate noise in a 3×3 grid.
    2. Eliminate noise in a 5×5 grid.
    3. Assign each element a state chosen from a set of predefined states.

The referenced paper clearly showcases the segmentation capabilities of the model. Yet, you might still ask yourself: *So what?*

One very relevant application can be found in the preprocessing of star images onboard spacecraft. The most accurate way for a satellite to determine its attitude (i.e. orientation) is by using star trackers. These are small optical sensors mounted on the satellite platform. Typically three units are used. Their primary purpose is the detection of star patterns, which assume fixed positions in an ICRF (International Celestial Reference Frame). By matching the perceived image to the known positions of these stars, the attitude of the spacecraft can be determined with arcsecond precision.

The entire pipeline from an image to the final attitude solution is a complex process. Therefore, we will focus only on the first step: segmenting the image and reducing noise levels.
"""

# ╔═╡ 8a7b7b90-bc6e-4cbf-bd66-a000acb56a92
md"""
## Loading the image
In a first step we will load the image (from within the repository) to Pluto. As we are not interested in the color components, we will transform the image to a grayscale and then to a matrix of floats.
"""

# ╔═╡ 8d5d3300-4cad-4adc-b865-1631a20a9ed6
begin
	star_image = load("./applications/img/star_image.png")
	(height, width) = size(star_image)
	plot(star_image, title="Star Image", size=(width,height), aspect_ratio=:equal)
end

# ╔═╡ 98417593-413e-465c-b46b-99e287fc24d4
begin
	star_image_gray = Gray.(star_image)
	star_image_array = Float64.(Array(star_image_gray))
end

# ╔═╡ 7f090510-017b-4f42-b3fa-43c67119f6b5
md"""
## Encoding the rules

Go over the refernced paper and extract the rules of the CA. You should perceive the pixel intensity as the physical phenomenon we are studying. For the sake of consistency, rescale the pixel intensities to have a value between 0 and 255.

### Rule 1

|    |    |    |
|----|----|----|
| $x_1$ | $x_2$ | $x_3$ |
| $x_4$ | $x_0$ | $x_5$ |
| $x_6$ | $x_7$ | $x_8$ |

We first introduce the helper function as follows:

``C(a, b) = \begin{cases} 1   & \text{if } a < b \\ -1  & \text{if } a > b \\ 0   & \text{if } a = b \\ \end{cases}``

This is then used to determine following value in the vicinity of the point ``x_0``, where ``k`` indicates the number of the iteration:


``M_{3 \times 3}(x_0) = \operatorname{sign} \left( \sum_{i=1}^8 C(x_0, x_i) \right)``

The value of ``M_{3 \times 3}(x_0)`` is then used to apply a simple rule:

``\begin{aligned} R_1: & \quad \text{If } M_{3 \times 3}(x_0) = 1 \quad \text{then}  \quad x_0(k+1) = x_0(k) + 1 \\ R_2: & \quad \text{If } M_{3 \times 3}(x_0) = -1 \quad \text{then} \quad x_0(k+1) = x_0(k) - 1 \\ R_3: & \quad \text{If } M_{3 \times 3}(x_0) = 0 \quad \text{then} \quad x_0(k+1) = x_0(k) \end{aligned}``

This step is applied *iter1* times.

### Rule 2

|       |       |       |       |        |
|-------|-------|-------|-------|--------|
| $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$  |
| $x_6$ | $x_7$ | $x_8$ | $x_9$ | $x_{10}$ |
| $x_{11}$ | $x_{12}$ | $\mathbf{x_0}$ | $x_{13}$ | $x_{14}$ |
| $x_{15}$ | $x_{16}$ | $x_{17}$ | $x_{18}$ | $x_{19}$ |
| $x_{20}$ | $x_{21}$ | $x_{22}$ | $x_{23}$ | $x_{24}$ |

Rule 2 is similar to rule 1, yet the size of the neighborhood around the cell is increased:

``M_{5\times 5}(x_0) = \operatorname{sign} \left( \sum_{i=1}^{24} C(x_0, x_i) \right)``

This step is applied *iter2* times.

### Rule 3

In a last step, the elements of the CA undergo a process that results in the identification of regions or objects characterized by homogeneous intensities and clearly defined edges. Each element is a assigned a final state. This is achieved through the division of the range of intensity levels or states into a smaller number of segments. The amount of segments can be chosen manually. We will call this value ``n``.

``x_0(k+1) = \text{round}\left(\frac{x_0(k)}{n}\right)``

"""

# ╔═╡ 7c680a64-e34a-46a2-ba09-26c415eeb57e
md"""
!!! tip "Assignment"
	- Write the functions necessary to apply stages 1, 2 and 3 of the cellular automaton.
	- Apply this segmentation to the provided image.
	- Compare the original image to the segmented image. What do you see? How do the parameters *iter1*, *iter2* and ``n`` impact your result?
	- What are the (dis)advantages of the approach cfr. the referenced paper?
"""

# ╔═╡ 9a369504-fdfe-46d3-969f-673580b47b99
md"""
# Phase transition in networks
## Generalities - Networks

The concept of graphs has been introduced in the lectures. In what follows we will give a brief recapitulation:

!!! info "Graph"
	Abstract mathematical structure consisting of vertices (or nodes) and edges (or links) connecting pairs of vertices. Representation of a real-world network (potentially simplified).
    
	Notation: ``G = (V, E)`` where ``V`` is the set of vertices and ``E`` is the set of edges.

In the following, we will only consider undirected networks i.e. if node $i$ is connected to node $j$, then node $j$ is automatically connected to node $i$ (as is the case with Facebook friendships). For unweighted networks of $N$ nodes, the network structure can be represented by an $N \times N$ adjacency matrix $A$:

```math
a_{i,j} = \left\{
                \begin{array}{ll}
                  1 & \text{if there is an edge from node $i$ to node $j$}\\
                  0 & \text{otherwise}\\
                \end{array}
                \right.
```
The degree of a node $i$ is the number of connections it has. In terms of the adjacency matrix
$A$, the degree of node $i$ is the sum of the i$^{\text{th}}$ row of $A$:
$$k_i = \sum_{j=1}^{N}a_{i,j}$$

The average node degree $\langle k \rangle$ is then given by: $$\langle k \rangle = \frac{1}{N} \sum_{i=1}^{N}k_{i}$$


## Erdös-Rényi random graph model
One of the simplest graph models is the Erdös-Rényi random graph model, denoted by $\mathcal{G}(N,p)$ with $N$ being the amount of nodes and $p$ being the probability that a link exists between two nodes. Self-loops are excluded. The value of $p$ is typically small (this is to avoid that the average degree $\langle k \rangle$ depends on $N$, cf. specialised literature for more details). When studying random graph model models, a major aim is to predict the average behaviour of certain network metrics and, if possible, their variance.



The Erdös-Rényi random graph  exhibits a phase transition. Let us consider the size (i.e., number of nodes) of the largest connected component in the network as a function of the mean degree ⟨k⟩. When ⟨k⟩ = 0, the network is trivially composed of N disconnected nodes. In the other extreme of ⟨k⟩ = N − 1, each node pair is adjacent such that the network is trivially connected. Between the two extremes, the network does not change smoothly in terms of the largest component size. Instead, a giant component, i.e., a component whose size is the largest and proportional to N, suddenly appears as ⟨k⟩ increases, marking a phase transition. The goal of this application is to determine this value by simulation.

## Problem solution
Resolve the problem and find the phase transition.
!!! tip "Subproblems"
	* generating a random graph
	* determine the average degree
	* identifying the size of the largest connected component
	* visualising the result
	* determine the critical value

"""

# ╔═╡ Cell order:
# ╟─dab1dfbe-5e52-11f0-05a0-df522fb51dd7
# ╠═cdecef07-e0f9-4372-b1c0-d284d763bb83
# ╟─be2db4dd-9ace-4985-8414-65124e48e7d3
# ╟─8a7b7b90-bc6e-4cbf-bd66-a000acb56a92
# ╠═8d5d3300-4cad-4adc-b865-1631a20a9ed6
# ╠═98417593-413e-465c-b46b-99e287fc24d4
# ╟─7f090510-017b-4f42-b3fa-43c67119f6b5
# ╟─7c680a64-e34a-46a2-ba09-26c415eeb57e
# ╟─9a369504-fdfe-46d3-969f-673580b47b99
