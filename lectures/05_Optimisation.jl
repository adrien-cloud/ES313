### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 05653593-4365-46a2-8e1b-79c3c8e11575
begin
	# Pkg needs to be used to force Pluto to use the current project instead of making an environment for each notebook
	using Pkg
	# this is redundant if you run it through start.jl, but to make sure...
	cd(joinpath(dirname(@__FILE__),".."))
    Pkg.activate(pwd())
	using PlutoUI
	PlutoUI.TableOfContents(depth=4)
end

# ╔═╡ b8671dde-ef61-4e07-8243-4d9521d2fcd0
# dependencies
begin
	using Plots 		# for normal plots
	using LaTeXStrings  # for fancy strings in plots 
	using NativeSVG     # for simple SVG plots 

	using JuMP 			# overal optimisation framework
	using GLPK 			# a versatile solver for linear programming problems
	using Tulip 		# an interior-point solver for linear programming problems

	using Optim 		# for global optimisation
	using Ipopt 		# for contrained optimisation
end

# ╔═╡ e0115026-3eb8-4127-b275-34c9013fed2c
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

# ╔═╡ 60e010dc-3669-424e-b001-085faa6b93ba
md"""# Optimisation
## Introduction
In a broad sense, *optimisation* refers to the process of making a system, design, or decision as effective or functional as possible. It involves finding the best solution from a set of feasible solutions. It typically be described in a more formal way by the following components:
- **Objective function**: The function that needs to be maximised or minimised.
- **Decision variables**: The variables that influence the outcome of the objective function.
- **Constraints**: Restrictions or limitations on the decision variables (if applicable).

You have already encountered this type of problems in the course "`ES122 - Vector Analysis, Fourier Analysis and Optimisation`". We will continue to build on the framework you have seen there, and always work with a minimisation problem. This allows us to write the formal optimisation problem as follows:
```math
\min_{\vec{x} \in \Omega } f(\vec{x}) \text{ subject to } \cases{\vec{h}(\vec{x}) = \vec{0}\\ \vec{g}(\vec{x}) \leq \vec{0}}
```

where 
```math
\begin{array}
\vec{x} \in \mathbb{R}^n,\\ 
\Omega \subseteq \mathbb{R}^n,\\  
f : \mathbb{R}^n \mapsto \mathbb{R},\\ 
\vec{h} :  \mathbb{R}^n \mapsto \mathbb{R}^m \; (m\leq n),\\
\vec{g} :  \mathbb{R}^n \mapsto \mathbb{R}^p.
\end{array}
```

We can classify optimisation as follows:
* Linear vs. Nonlinear
  - Linear optimisation: the objective function and constraints are linear functions of the decision variables. *Example*: maximising profit in a production process where both profit and constraints (e.g., resource limitations) are linear.
  - Nonlinear optimisation: the objective function or constraints (or both) are nonlinear. *Example* optimising the shape of an aircraft wing where the relationship between its shape and performance is nonlinear.

* Unconstrained vs. Constrained Optimisation
  - Unconstrained optimisation: no restrictions are placed on the decision variables. *Example*: finding the minimum of a simple quadratic function.
  - Constrained Optimisation: the decision variables must satisfy certain constraints. *Example*: minimising the cost of production subject to constraints on material availability and production capacity.



Optimisation is a critical tool used across various fields to improve efficiency and outcomes. Being able to formulate a real-world application as an optimisation problem and knowing what is the most appropriate technique to solve it, is a very valuable and an indispensable skill for engineers.

"""

# ╔═╡ b02cadd0-65d0-4395-ba64-35f023696ce7
md"""
## Linear programming
### Definition

Formally, a linear program is an optimization problem of the form:

```math
\min_{\vec{x} \in \Omega } \vec{c}^\mathsf{T}\vec{x} \; \text{ subject to } 
\cases{
\mathbf{A}\vec x=\vec b\\
\vec x\ge\vec 0}
```

where $\vec c\in\mathbb R^n$, $\vec b\in\mathbb R^m$ and $\mathbf A \in \mathbb R^{m\times n}$. 
The name "linear programming" stems from the contraints and the objective function being represented by linear relationships, so it can be considered as a special case of the more general family of optimisation problems. 
The vector inequality $\vec x\ge\vec 0$ means that each component of $\vec x$ is nonnegative. Several variations of this problem are possible; e.g. instead of minimizing, we can maximize, or the constraints may be in the form of inequalities, such as $\mathbf A\vec x\ge \vec b$ or $\mathbf A\vec x\le\vec b$. These variations can all be rewritten into the standard form.

### Example
```math
\min_{\vec{x}} x_1 +5 x_2 \; \text{ subject to } 
\cases{
5x_1 + 6x_2 \leq 30\\
2x_1 + 2x_2 \leq 12\\
x_1 \ge 0\\
x_2 \ge 0}
```
This problem can be represented graphically (see figure below), and we can identify the feasible region (green) that imposes the different constraints. We also show two instances of the objective function. An optimal solution can be obtained by finding the intersection of a parallel line with the feasible region's boundary. This feasible region is sometimes referred to as a simplex. A solution will be a corner point or edge of the simplex. The extension to more than two dimensions is detailed below.
"""


# ╔═╡ 7a9fa9f1-b50b-4e08-b5c3-537226e3deda
let
	x = -2:6
	plot(x, (30 .- 5 .* x) ./ 6, linestyle=:dash, label=L"5x_1+6x_2=30")
	plot!(x, (12 .- 3 .* x) ./ 2, linestyle=:dash, label=L"3x_1+2x_2=12")
	plot!([0,4,1.5,0,0],[0,0,3.75,5,0], linewidth=2, label="constraints")
	plot!(x, -x ./ 5, line=:dash, color=:black, label=L"f\left(x_1,x_2\right)=x_1+5x_2=0")
	plot!(x, (25 .- x) ./ 5, line=:dot, color=:black, label=L"f\left(x_1,x_2\right)=x_1+5x_2=25")
end

# ╔═╡ 88447980-da25-404f-abcf-29927d3c0168
md"""
### Non-standard forms
Theorems and solution techniques are usually stated for problems in the standard form. Other forms of linear programs can be converted to the standard form, as illustrated below.

**Non-standard form 1:**
```math
\min_{\vec{x} \in \Omega } \vec{c}^\mathsf{T}\vec{x} \; \text{ subject to } 
\cases{
\mathbf{A}\vec{x} \ge \vec{b}\\
\vec{x} \ge \vec{0}}
```
We can introduce so-called *surplus variables* ( $\vec{y}$ ) allowing to write the problem in standard form:
```math
\min_{\vec{x} \in \Omega } \vec{c}^\mathsf{T}\vec{x} \; \text{ subject to } 
\cases{
\mathbf{A}\vec x - \mathbf{I}\vec{y} =  \vec b\\
\vec{x}\ge\vec 0\\
\vec{y}\ge\vec 0}
```
where $\mathbf{I}$ denotes the identity matrix.

**Non-standard form 2:**

In a similar fashion, we can transform the following problem
```math
\min_{\vec{x} \in \Omega } \vec{c}^\mathsf{T}\vec{x} \; \text{ subject to } 
\cases{
\mathbf{A}\vec{x} \leq \vec{b}\\
\vec{x} \ge \vec{0}}
```
by introducing the so-called *slack variables* ( $\vec{y}$ ) allowing to write the problem in standard form:
```math
\min_{\vec{x} \in \Omega } \vec{c}^\mathsf{T}\vec{x} \; \text{ subject to } 
\cases{
\mathbf{A}\vec x + \mathbf{I}\vec{y} =  \vec b\\
\vec{x}\ge\vec 0\\
\vec{y}\ge\vec 0}
```

#### Example
Consider the following optimization problem:

```math

\max x_2-x_1 \; \text{ subject to }
\begin{cases}
3x_1 = x_2-5 \\
\left |x_2 \right| \le 2\\
x_1 \le 0
\end{cases}
```

To convert the problem into a standard form, we perform the following steps:

1. Change the objective function to:

```math
\min x_1 - x_2
```

2. Substitute $x_1=-x_1^\prime$ (because in standard form, our variables should be positive).

3. Write $\left|x_2\right|\le2$ as $x_2\le 2$ and $-x_2\le 2$.

4. Introduce slack variables $y_1$ and $y_2$, and convert the inequalities above to

```math
\begin{cases}
- x_2 + y_1 = 2\\
- x_2 + y_2 = 2
\end{cases}
```

5. Write $x_2=u-v$ with $u,v\ge0$ (again, because in standard form, our variables should be positive).

Hence, we obtain

```math
\min -x_1^\prime-u+v \; \textrm{ subject to }
\begin{cases}
3x_1^\prime+u-v=5 \\
u-v+y_1=2 \\
v-u+y_2=2 \\
x_1^\prime,u,v,y_1,y_2\ge0
\end{cases}
```
"""

# ╔═╡ 94ac3d2e-6f8f-45ef-b9ba-93b3af11098a
md"""### Fundamental theorem of linear programming

We consider the system of equalities

```math
\mathbf{A}\vec x=\vec b
```

where $\mathrm{rank}\,\mathbf A=m$.

Let $\mathbf B$ a square matrix whose columns are $m$ linearly independent columns of $\mathbf A$. If necessary, we reorder the columns of $\mathbf A$ so that the columns in $\mathbf B$ appear first: $\mathbf A$ has the form $\left(\mathbf B |\mathbf N\right)$.

The matrix is nonsingular, and thus we can solve the equation

```math
\mathbf B\vec x_\mathbf B = \vec b
```

The solution is $\vec x_\mathbf B = \mathbf B^{-1}\vec b$.

Let $\vec x$ be the vector whose first $m$ components are equal to $\vec x_\mathbf B$ and the remaining components are equal to zero. Then $\vec x$ is a solution to $\mathbf A\vec x=\vec b$. We call $\vec x$  a _basic solution_. Its components refering to the the components of $\vec x_\mathbf B$ are called _basic variables_.

- If some of the basic variables are zero, then the basic solution is _degenerate_.
- A vector $\vec x$ satisfying $\mathbf A\vec x=\vec b$, $\vec x \ge \vec 0$, is said to be a _feasible solution_.
- A feasible solution that is also basic is called a _basic feasible solution_.

The fundamental theorem of linear programming states that when solving a linear programming problem, we need only consider basic feasible solutions. This is because the optimal value (if it exists) is always achieved at a basic solution."""

# ╔═╡ 163c7e36-7394-4550-92a6-5a263cf69e09
md"""### Simplex method for solving linear programming problems
The simplex method belongs to a general class of algorithms for constrained optimization known as _active set methods_, which explicitly maintain estimates of the active and inactive index sets that are updated at each step of the algorithm. Like most active set methods, the simplex method makes only modest changes to these index sets at each step; a single index is exchanged between $\mathcal B$ into $\mathcal N$.

One undesirable feature of the simplex method attracted attention from its earliest days. Though highly efficient on almost all practical problems (the method generally requires at most $2m$ to $3m$ iterations, where $m$ is the row dimension of the constraint matrix, there are pathological problems on which the algorithm performs very poorly. The complexity of the simplex method is _exponential_, roughly speaking, its running time may be an exponential function of the dimension of
the problem.


#### Karush-Kuhn-Tucker Conditions

We consider the linear program

```math
\begin{aligned}
\min &\vec{c}^\mathsf{T}\vec x\\
\textrm{ subject to } &\begin{cases}
\mathbf{A}\vec x=\vec b\\
\vec x\ge\vec 0
\end{cases}
\end{aligned}
```

and define the Lagrangian function

```math
\mathcal L\left(\vec x,\vec \lambda,\vec s\right) = \vec{c}^\mathsf{T}\vec x - \vec \lambda^\mathsf{T}\left(\mathbf A\vec x-\vec b\right) - \vec s^\mathsf{T}\vec x
```

where $\vec \lambda$ are the multipliers for the equality constraints and $\vec s$ are the multipliers for the bound constraints.

The Karush-Kuhn-Tucker condition states that to find the first-order necessary conditions for $\vec x^\star$ to be a solution of the problem, their exist $\vec \lambda^\star$ and $\vec s^\star$ such that

```math
\begin{aligned}
\mathbf A^\mathsf{T}\vec \lambda^\star+\vec s^\star&=\vec c\\
\mathbf A\vec x^\star&=\vec b\\
\vec x^\star&\ge\vec 0\\
\vec s^\star&\ge \vec 0\\
\left(\vec x^\star\right)^\mathsf{T}\vec s^\star&=0
\end{aligned}
```

The first eqation states that the gradient of the Lagrangian with respect to $\vec x$ must be zero and the last equation that at least $x_i$ or $s_i$ must be zero for each $i=1,2,\dots,n$.

It can be shown that these conditions are also sufficient (cf. ES122).
"""

# ╔═╡ b67154e3-256e-44e2-be84-ac3f2462d58a
md"""
#### Basic principle
All iterates of the simplex method are basic feasible points and therefore vertices of the feasible polytope. Most steps consists of a move from one vertex to an adjacent one. On most steps (but not all), the value of the objective function $\vec{c}^\mathsf{T}\vec x$ is decreased. Another type of step occurs when the problem is unbounded: the step is an edge along which the objective funtion is reduced, and along which we can move infinitely far without reaching a vertex.

The major issue at each simplex iteration is to decide which index to remove from the basic index set $\mathcal B$. Unless the step is a direction of unboundness, a single index must be removed from $\mathcal B$ and replaced by another from outside $\mathcal B$. We can gain some insight into how this decision is made by looking again at the KKT conditions.


First, define the nonbasic index set $\mathcal N = \left\{1,2,\dots,n\right\} \setminus \mathcal B$. Just as $\mathbf B$ is the basic matrix, whose columns are $\mathbf A_i$ for $i\in\mathcal B$, we use $\mathbf N$ to denote the nonbasic matrix $\mathbf N=\left[\mathbf A_i\right]_{i\in\mathcal N}$. We also partition the vectors $\vec x$, $\vec s$ and $\vec c$ according to the index sets $\mathcal B$  and $\mathcal N$, using the notation

```math
\begin{aligned}
\vec x_\mathbf B=\left[\vec x_i \right]_{i\in\mathcal B},&\qquad\vec x_\mathbf N=\left[\vec x_i \right]_{i\in\mathcal N}\\
\vec s_\mathbf B=\left[\vec s_i \right]_{i\in\mathcal B},&\qquad\vec s_\mathbf N=\left[\vec s_i \right]_{i\in\mathcal N}\\
\vec c_\mathbf B=\left[\vec c_i \right]_{i\in\mathcal B},&\qquad\vec c_\mathbf N=\left[\vec c_i \right]_{i\in\mathcal N}
\end{aligned}
```

From the second KKT conditions, we have that

```math
\mathbf A \vec x= \mathbf B \vec x_\mathbf B + \mathbf N \vec x_\mathbf N=\vec b\,.
```

The _primal_ variable $\vec x$ for this simplex iterate is defined as

```math
\vec x_\mathbf B = \mathbf B^{-1}\vec b,\qquad \vec x_\mathbf N=\vec 0\,.
```

Since we are dealing only with basic feasible points, we know that $\mathbf B$ is nonsingular and that $\vec x_\mathbf B\ge\vec0$, so this choice of $\vec x$ satisfies two of the KKT conditions.

We choose $\vec s$ to satisfy the complimentary condition (the last one) by setting $\vec s_\mathbf B=\vec 0$. The remaining components $\vec \lambda$ and $\vec s_\mathbf N$ can be found by partitioning this condition into $\vec c_\mathbf B$ and $\vec c_\mathbf N$ components and using $\vec s_\mathbf B=\vec 0$ to obtain

```math
\mathbf B^\mathsf{T}\vec \lambda=\vec c_\mathbf B,\qquad \vec N^\mathsf{T}\vec\lambda+\vec s_\mathbf N = \vec c_\mathbf N\,.
```

Since $\mathbf B$ is square and nonsingular, the first equation uniquely defines $\vec \lambda$ as

```math
\vec \lambda = \left(\mathbf B^\mathsf{T}\right)^{-1}\vec c_\mathbf B\,.
```

The second equation implies a value for $\vec s_\mathbf N$:

```math
\vec s_\mathbf N = \vec c_\mathbf N - \mathbf N^\mathsf{T}\vec \lambda=\vec c_\mathbf N -\left(\mathbf B ^{-1}\mathbf N\right)^\mathsf{T}\vec c_\mathbf B\,.
```

Computation of the vector $\vec s_\mathbf N$ is often referred to as _pricing_. The components of $\vec s_\mathbf N$ are often called the _reduced costs_ of the nonbasic variables $\vec x_\mathbf N$.

The only KKT condition that we have not enforced explicitly is the nonnegativity condition $\vec s \ge \vec 0$. The basic components $\vec s_\mathbf B$ certainly satisfy this condition, by our choice $\vec s_\mathbf B = 0$. If the vector $\vec s_\mathbf N$ also satisfies $\vec s_\mathbf N \ge \vec 0$, we have found an optimal
vector triple $\left(\vec x^\star, \vec \lambda^\star, \vec s^\star\right)$, so the algorithm can terminate and declare success. Usually, however, one or more of the components of $\vec s_\mathbf N$ are negative. The new index to enter the basis index set $\mathcal B$ is chosen to be one of the indices $q \in \mathcal N$ for which $s_q < 0$. As we show below, the objective $\vec{c}^\mathsf{T}\vec x$ will decrease when we allow $x_q$ to become positive if and only if 

1. ``s_q < 0`` and
2. it is possible to increase $x_q$ away from zero while maintaining feasibility of $\vec x$.

Our procedure for altering $\mathcal B$ and changing $\vec x$ and $\vec s$ can be described accordingly as follows:

- allow $x_q$ to increase from zero during the next step;
- fix all other components of $\vec x_\mathbf N$ at zero, and figure out the effect of increasing $x_q$ on the current basic vector $\vec x_\mathbf B$, given that we want to stay feasible with respect to the equality constraints $\mathbf{A}\vec x=\vec b$;
- keep increasing $x_q$ until one of the components of $\vec x_\mathbf B$ ($x_p$, say) is driven to zero, or determining that no such component exists (the unbounded case);
- remove index $p$ (known as the leaving index) from $\mathcal B$ and replace it with the entering index $q$.

This process of selecting entering and leaving indices, and performing the algebraic operations necessary to keep track of the values of the variables $\vec x$, $\vec \lambda$, and $\vec s$, is sometimes known as _pivoting_.

We now formalize the pivoting procedure in algebraic terms. Since both the new iterate $\vec x^+$ and the current iterate $\vec x$ should satisfy $\mathbf A\vec x=\vec b$, and since $\vec x_\mathbf N=\vec 0$ and $\vec x_i^{+}=0$ for $i\in\mathcal N\setminus\left\{q\right\}$ we have

```math
\mathbf A\vec x^+=\mathbf B\vec x_\mathbf B^+ +\vec A_q x_q^+=\vec b=\mathbf B\vec x_\mathbf B=\mathbf A\vec x\,.
```

By multiplying this expression by $\mathbf B^{-1}$ and rearranging, we obtain

```math
\vec x_\mathbf B^+=\vec x_\mathbf B-\mathbf B^{-1}\vec A_q x_q^+
```

Geometrically speaking, we move along an edge of the feasible polytope that decreases $\vec{c}^\mathsf{T}\vec x$. We continue to move along this edge until a new vertex is encountered. At this vertex, a new constraint $x_p \ge 0$ must have become active, that is, one of the components $x_p$, $p \in \mathbf B$, has decreased to zero. We then remove this index $p$ from the basis index set $\mathcal B$ and replace it by $q$.

It is possible that we can increase $x_q^+$ to $\infty$ without ever encountering a new vertex. In other words, the constraint $x_\mathbf B^+=x_\mathbf B-\mathbf B^{-1}\vec A_q\vec x_q^+\ge 0$ holds for all positive values of $x_q+$. When this happens, the linear program is unbounded; the simplex method has identified a
ray that lies entirely within the feasible polytope along which the objective $\vec{c}^\mathsf{T}\vec x$ decreases to $−\infty$.
"""

# ╔═╡ d75a7f24-e407-4585-b9ce-0e2d13e0dba1
md"""
#### Algorithmic approach
Given $\mathcal B$, $\mathcal N$, $x_\mathbf B=\mathbf B^{-1}\vec b\ge 0$,$\vec x_\mathbf N=\vec 0$;

1. Solve $\vec \lambda = \left(\mathbf B^\mathsf{T}\right)^{-1}\vec c_\mathbf B$, and  compute $\vec s_\mathbf N =\vec c_\mathbf N - \mathbf N^\mathsf{T} \vec \lambda$ (pricing);
2. If $\vec s_\mathbf N \ge \vec 0$ stop (optimal point found);
3. Select $q\in\mathcal N$ with $s_q<0$ and solve $\vec d=\mathbf B^{-1}\vec A_q$;
4. If $\vec d\le\vec 0$ stop (problem is unbounded);
5. Calculate $x_q^+=\min_{i|d_i>0}\frac{\left(x_\mathbf B\right)_i}{d_i}$, and use $p$ to denote the minimizing $i$;
6. Update $\vec x_\mathbf B^+=\vec x_\mathbf B-x_q^+\vec d$;
7. Change $\mathcal B$ by adding $q$ and removing the basic variable corresponding to column $p$ of $\mathbf B$.
"""

# ╔═╡ 5ac0b0d5-ce4d-4851-9691-c557e5b8fb90
md"""
#### Example
Consider the following problem:
```math
\min -4x_1 - 2x_2 \; \text{ subject to } \; 
\cases{
x_1 + x_2 \le 5 \\
2x_1 + x_2/2 \le 8 \\
x_1 \ge 0 \\
x_2 \ge 0
}
```

Suppose we start with the basis index set $\mathcal B=\left\{3,4\right\}$, for which we have the following values:
"""

# ╔═╡ a6ff71ff-522f-4f41-bb2c-edd8fc944024
let B = [1 0;0 1], N = [1 1;2 0.5], b = [5;8], cB = [0;0], cN = [-4;-2]
	xB = inv(B)*b
	λ = inv(transpose(B))*cB
	sN = cN - transpose(N)*λ
	xB, λ, sN
end

# ╔═╡ a95111dd-27c8-4d66-a278-ca58badf1111
md"""```math
\vec x_\mathbf B =
\begin{pmatrix}
x_3\\
x_4
\end{pmatrix}
= 
\begin{pmatrix}
5\\
8
\end{pmatrix}
\,\qquad\vec\lambda = 
\begin{pmatrix}
0\\
0
\end{pmatrix}
\,\qquad\vec s_\mathbf N =
\begin{pmatrix}
s_1\\
s_2
\end{pmatrix}
=
\begin{pmatrix}
-4\\
-2
\end{pmatrix}
```

and an objective value of $\vec{c}^\mathsf{T}\vec x=0$. Since both elements of $\vec s_\mathbf N$ are negative, we could choose either 1 or 2 to be the entering variable. Suppose we choose $q=1$. We obtain

```math
\vec d = \begin{pmatrix}
1\\
2
\end{pmatrix}\,,
```

so we cannot (yet) conclude that the problem is unbounded. By performing the ratio calculation, we find that $p=2$ (corresponding to index 4) and $x_1^+=4$."""

# ╔═╡ 167d22c7-a625-4dc3-98cd-fcf8a1789c25
let B = [1 0;0 1], N = [1 1;2 0.5], b = [5;8], cB = [0;0], cN = [-4;-2], xB = [5;8]
	q = 1
	Aq = [1;2]
	d = inv(B)*Aq
	ratio = xB./d
	xq = minimum(ratio)
	xB -= d * xq
	d, ratio, xq, xB
end

# ╔═╡ 6ff21901-4055-4221-88ed-f1a3aec2f2ce
md"""We update the basic and nonbasic index sets to $\mathcal B=\left\{3,1\right\}$ and  $\mathcal N=\left\{4,2\right\}$, and move to the next iteration.

At the second iteration, we have"""

# ╔═╡ bb7ebc81-676d-41a9-b119-95b84655378b
let B = [1 1;0 2], N = [0 1;1 0.5], b = [5;8], cB = [0;-4], cN = [0;-2]
	xB = inv(B)*b
	λ = inv(transpose(B))*cB
	sN = cN - transpose(N)*λ
	xB, λ, sN
end

# ╔═╡ 7da441d0-3a9c-4ce0-80f1-508f8da944f6
md"""
```math
\vec x_\mathbf B =
\begin{pmatrix}
x_3\\
x_1
\end{pmatrix}
= 
\begin{pmatrix}
1\\
4
\end{pmatrix}
\,\qquad\vec\lambda = 
\begin{pmatrix}
0\\
-2
\end{pmatrix}
\,\qquad\vec s_\mathbf N =
\begin{pmatrix}
s_4\\
s_2
\end{pmatrix}
=
\begin{pmatrix}
2\\
-1
\end{pmatrix}
```

with an objective value of -16. We see that $\vec s_\mathbf N$ has one negative component, corresponding to the index $q=2$, se we select this index to enter the basis. We obtain
"""

# ╔═╡ fa47ae35-f707-494c-8fb6-ef65076f13c4
let B = [1 1;0 2], N = [0 1;1 0.5], b = [5;8], cB = [0;-4], cN = [0;-2], xB=[1;4]
	q = 2
	Aq = [1;0.5]
	d = inv(B)*Aq
	ratio = xB./d
	xq = minimum(ratio)
	xB -= d * xq
	d, ratio, xq, xB
end

# ╔═╡ 5349edce-28c7-4f65-a293-2a7059df2e60
md"""```math
\vec d = \begin{pmatrix}
\frac{3}{4}\\
\frac{1}{4}
\end{pmatrix}\,,
```

so again we do not detect unboundedness. Continuing, we find that the minimum value of $x_2^+$ is $\frac{4}{3}$, and that $p=1$, which indicates that index 3 will leave the basic index set $\mathcal B$. We update the index sets to $\mathcal B=\left\{2,1\right\}$ and  $\mathcal N=\left\{4,3\right\}$ and continue.

At the start of the third iteration, we have
"""

# ╔═╡ 7f2110fa-a514-421e-9d13-57400f56c63b
let B = [1 1;0.5 2], N = [0 1;1 0], b = [5;8], cB = [-2;-4], cN = [0;0]
	xB = inv(B)*b
	λ = inv(transpose(B))*cB
	sN = cN - transpose(N)*λ
	xB, λ, sN
end

# ╔═╡ e1a856c5-2b03-47aa-87f0-7fcb952cc2e3
md"""```math
\vec x_\mathbf B =
\begin{pmatrix}
x_2\\
x_1
\end{pmatrix}
= 
\begin{pmatrix}
\frac{4}{3}\\
\frac{11}{3}
\end{pmatrix}
\,\qquad\vec\lambda = 
\begin{pmatrix}
-\frac{4}{3}\\
-\frac{4}{3}
\end{pmatrix}
\,\qquad\vec s_\mathbf N =
\begin{pmatrix}
s_4\\
s_3
\end{pmatrix}
=
\begin{pmatrix}
\frac{4}{3}\\
\frac{4}{3}
\end{pmatrix}
```

with an objective value of $-\frac{52}{3}$. We see that $\vec s_\mathbf N\ge\vec 0$, so the optimality test is satisfied, and we terminate.

We need to flesh out this procedure with specifics of three important aspects of the implementation:

- Linear algebra issues—maintaining an LU factorization of $\mathbf B$ that can be used to solve for $\vec \lambda$ and $\vec d$.
- Selection of the entering index $q$ from among the negative components of $\vec s_\mathbf N$. (In general, there are many such components.)
- Handling of degenerate bases and degenerate steps, in which it is not possible to choose a positive value of $x_q$ without violating feasibility.

Proper handling of these issues is crucial to the efficiency of a simplex implementation. We will use a software package to handle these details."""

# ╔═╡ 8cb2b3ae-1ebf-4694-adf1-42c1c7863016
md"""
#### Implementations
In Julia, the [`JuMP package`](https://jump.dev/JuMP.jl/stable/) is a domain-specific modeling language for mathematical optimization. It supports different open-source and commercial solvers for a variety of problem classes, including linear, mixed-integer, second-order conic, semidefinite, and nonlinear programming. This allows you to use the same overal interface to solve different kinds of problems or even to try different solvers on the same problem without having to change the problem's syntax.

The [GLPK solver](https://www.gnu.org/software/glpk/) is an open-source solver that can solve large-scale linear programming problems. It can use the simplex method (among others) that was mentioned before. In Julia, this solver is made available through the [`GLPK.jl`](https://github.com/jump-dev/GLPK.jl) package, which simply acts as a wrapper for the GLPK solver (which is written in C).

Below we use it to solve the following optimisation problem:
```math
\min -4x_1 - 2x_2 \; \text{ subject to } \; 
\cases{
x_1 + x_2 \le 5 \\
2x_1 + x_2/2 \le 8 \\
x_1 \ge 0 \\
x_2 \ge 0
}
```

"""

# ╔═╡ b4ed8968-fc9d-4f98-8efd-656a6a0a7acf
md"In the example you will see that you don't need to transform the problem into the standard form, JuMP takes care of that for you."

# ╔═╡ 3c5b411c-872f-4d67-b307-d686d649b129
let
	# define a JuMP model that will use the GLPK optimizer, which by default uses the simplex method
	model = Model(GLPK.Optimizer)
	# define our variables
	@variable(model, x1 >= 0)
	@variable(model, x2 >= 0)
	# define the objective function
	@objective(model, Min, -4*x1 -2*x2)
	# add the constraints
	@constraint(model, con1, x1 + x2 <= 5)
	@constraint(model, con2, 2*x1 + 0.5*x2 <= 8)
	# show the model
	@info "Our model:\n $(model)"
	# determine the solution
	optimize!(model)
	# why did the solver stop?
	@info termination_status(model)
	# what is the objective function's value?
	@info "Objective function value: $(objective_value(model))"
	# show the solution
	@info "Solution:\n - x₁ : $(value(x1))\n - x₂ : $(value(x2))	"
end

# ╔═╡ 98be776a-108c-4bc5-9095-30e8756ff048
md"We can still use the graphical representation to check our answer. More elaborated examples are given in the applications."

# ╔═╡ a99cffd7-d37c-47b2-82c2-f57cf90f3d95
let
	x = -1:5
	plot(x, 5 .- x, linestyle=:dash, label=L"x_1+x_2=5")
	plot!(x, (8 .- 2 .* x) ./ 0.5, linestyle=:dash, label=L"2x_1+0.5x_2=8")
	plot!([0,4,11/3,0,0],[0,0,4/3,5,0], linewidth=2, label="constraints")
	plot!(x, -4 .* x ./ 2, label=L"f\left(x_1,x_2\right)=-4x_1-2x_2=0")
	plot!(x, (-16 .+ 4 .* x) ./ -2, label=L"f\left(x_1,x_2\right)=-4x_1-2x_2=-16")
	plot!(x, (-52/3 .+ 4 .* x) ./ -2, label=L"f\left(x_1,x_2\right)=-4x_1-2x_2=-52/3")
end

# ╔═╡ 060516fe-0b94-4278-9f81-913d336c67d1
md"""
### Interiod Point Method for solving linear programming problems

In the 1980s it was discovered that many large linear programs could be solved efficiently by using formulations and algorithms from nonlinear programming and nonlinear equations. One characteristic of these methods was that they required all iterates to satisfy the inequality constraints in the problem _strictly_, so they became known as interior-point methods. By the early 1990s, a subclass of interior-point methods known as primal-dual methods had distinguished themselves as the most efficient practical approaches, and proved to be strong competitors to the simplex method on large problems.

Interior-point methods arose from the search for algorithms with better theoretical properties than the simplex method. The simplex method can be inefficient on certain pathological problems. Roughly speaking, the time required to solve a linear program may be exponential in the size of the problem, as measured by the number
of unknowns and the amount of storage needed for the problem data. For almost all practical problems, the simplex method is much more efficient than this bound would suggest, but its poor worst-case complexity motivated the development of new algorithms with better guaranteed performance.

Interior-point methods share common features that distinguish them from the simplex method. Each interior-point iteration is expensive to compute and can make significant progress towards the solution, while the simplex method usually requires a larger number of inexpensive iterations. Geometrically speaking, the simplex method works its way around the boundary of the feasible polytope, testing a sequence of vertices in turn until it finds the optimal one. Interior-point methods approach the boundary of the feasible set only in the limit. They may approach the solution either from the interior or the exterior of the feasible region, but they never actually lie on the boundary of this region.

#### Implementations
The Julia package [`Tulip`](https://github.com/ds4dm/Tulip.jl) is an interior-point solver that can be used for linear programming problems. It is typically used within the broader framework of JuMP. Below we illustrate how to use this to solve the following problem
```math
\min -4x_1 - 2x_2 \; \text{ subject to } \; 
\cases{
x_1 + x_2 \le 5 \\
2x_1 + x_2/2 \le 8 \\
x_1 \ge 0 \\
x_2 \ge 0
}
```
"""

# ╔═╡ b0601aae-10ff-4405-b498-6649a0036e1e
let
	# define a JuMP model that will use the Tulip optimizer
	model = Model(Tulip.Optimizer)
	# define our variables
	@variable(model, x1 >= 0)
	@variable(model, x2 >= 0)
	# define the objective function
	@objective(model, Min, -4*x1 -2*x2)
	# add the constraints
	@constraint(model, con1, x1 + x2 <= 5)
	@constraint(model, con2, 2*x1 + 0.5*x2 <= 8)
	# show the model
	@info "Our model:\n $(model)"
	# determine the solution
	optimize!(model)
	# show the solution
	@info "Solution:\n - x₁ : $(value(x1))\n - x₂ : $(value(x2))	"
end

# ╔═╡ d07169d1-3605-449e-a3fd-c2646e876da9
md"""
### Applications
#### Economy

A manufacturer produces  four different  products  $X_1$, $X_2$, $X_3$ and $X_4$. There are three inputs to this production process:

- labor in man weeks,  
- kilograms of raw material A, and 
- boxes  of raw  material  B.

Each product has different input requirements. In determining each  week's production schedule, the manufacturer cannot use more than the available amounts of  manpower and the two raw  materials:

|Inputs|$X_1$|$X_2$|$X_3$|$X_4$|Availabilities|
|------|-----|-----|-----|-----|--------------|
|Person-weeks|1|2|1|2|20|
|Kilograms of material A|6|5|3|2|100|
|Boxes of material B|3|4|9|12|75|
|Production level|$x_1$|$x_2$|$x_3$|$x_4$| |

These constraints can be written in mathematical form

```math
\begin{aligned}
x_1+2x_2+x_3+2x_4\le&20\\
6x_1+5x_2+3x_3+2x_4\le&100\\
3x_1+4x_2+9x_3+12x_4\le&75
\end{aligned}
```

Because negative production levels are not meaningful, we must impose the following nonnegativity constraints on the production levels:

```math
x_i\ge0,\qquad i=1,2,3,4
```

Now suppose that one unit of product $X_1$ sells for €6 and $X_2$, $X_3$ and $X_4$ sell for €4, €7 and €5, respectively. Then, the total revenue for any production decision $\left(x_1,x_2,x_3,x_4\right)$ is

```math
f\left(x_1,x_2,x_3,x_4\right)=6x_1+4x_2+7x_3+5x_4
```

The problem is then to maximize $f$ subject to the given constraints."""

# ╔═╡ f40dced6-78c6-4ad0-88b1-85896573f330
let
	model = Model(GLPK.Optimizer)
	@variable(model, 0 <= x1)
	@variable(model, 0 <= x2)
	@variable(model, 0 <= x3)
	@variable(model, 0 <= x4)
	@objective(model, Max, 6*x1 + 4*x2 + 7*x3 + 5*x4)
	@constraint(model, con1,   x1 + 2*x2 +   x3 +  2*x4 <= 20)
	@constraint(model, con2, 6*x1 + 5*x2 + 3*x3 +  2*x4 <= 100)
	@constraint(model, con3, 3*x1 + 4*x2 + 9*x3 + 12*x4 <= 75)
	optimize!(model)
	termination_status(model), primal_status(model), [value(x1), value(x2), value(x3), value(x4)], objective_value(model)
end

# ╔═╡ ed99bd93-bb76-4f18-af4b-c3439b40d3e0
md"""#### Manufacturing

A manufacturer produces two different products $X_1$ and $X_2$ using three machines $M_1$, $M_2$, and $M_3$. Each machine can be used only for a limited amount of time. Production times of each product on each machine are given by 

|Machine|Production time $X_1$|Production time $X_2$|Available time|
|-------|---------------------|---------------------|--------------|
|$M_1$  |1                    |1                    |8             |
|$M_2$  |1                    |3                    |18            |
|$M_3$  |2                    |1                    |14            |
|Total  |4                    |5                    |              |

The objective is to maximize the combined time of utilization of all three machines.

Every production decision must satisfy the constraints on the available time. These restrictions can be written down using data from the table.

```math
\begin{aligned}
x_1+x_2&\le8\,,\\
x_1+3x_2&\le18\,,\\
2x_1+x_2&\le14\,,
\end{aligned}
```

where $x_1$ and $x_2$ denote the production levels. The combined production time of all three machines is

```math
f\left(x_1,x_2\right)=4x_1+5x_2\,.
```"""

# ╔═╡ 7dcb0c9a-8036-4424-ab97-d7a909dbcb02
let
	model = Model(GLPK.Optimizer)
	@variable(model, 0 <= x1)
	@variable(model, 0 <= x2)
	@objective(model, Max, 4*x1 + 5*x2)
	@constraint(model, con1,   x1 +   x2 <= 8)
	@constraint(model, con2,   x1 + 3*x2 <= 18)
	@constraint(model, con3, 2*x1 +   x2 <= 14)
	optimize!(model)
	termination_status(model), primal_status(model), [value(x1), value(x2)], objective_value(model)
end

# ╔═╡ 6a727c57-e605-4813-aef8-b23766daacda
md"""#### Transportation

A manufacturing company has plants in cities A, B, and C. The company produces and distributes its product to dealers in various cities. On a particular day, the company has 30 units of its product in A, 40 in B, and 30 in C. The company plans to ship 20 units to D, 20 to E, 25 to F, and 35 to G, following orders received from dealers. The transportation costs per unit of each product between the cities are given by

|From|To D|To E|To F|To G|Supply|
|----|----|----|----|----|------|
|A   |7   |10  |14  |8   |30    |
|B   |7   |11  |12  |6   |40    |
|C   |5   |8   |15  |9   |30    |
|Demand|20|20  |25  |35  |100   |

In the table, the quantities supplied and demanded appear at the right and along the bottom of the table. The quantities to be transported from the plants to different destinations are represented by the decision variables.

This problem can be stated in the form:

```math
\min 7x_{AD}+10x_{AE}+14x_{AF}+8x_{AG}+7x_{BD}+11x_{BE}+12x_{BF}+6x_{BG}+5x_{CD}+8x_{CE}+15x_{CF}+9x_{CG}
```

subject to

```math
\begin{aligned}
x_{AD}+x_{AE}+x_{AF}+x_{AG}&=30\\
x_{BD}+x_{BE}+x_{BF}+x_{BG}&=40\\
x_{CD}+x_{CE}+x_{CF}+x_{CG}&=30\\
x_{AD}+x_{BD}+x_{CD}&=20\\
x_{AE}+x_{BE}+x_{CE}&=20\\
x_{AF}+x_{BF}+x_{CF}&=25\\
x_{AG}+x_{BG}+x_{CG}&=35
\end{aligned}
```

In this problem, one of the constraint equations is redundant because it can be derived from the rest of the constraint equations. The mathematical formulation of the transportation problem is then in a linear programming form with twelve (3x4) decision variables and six (3 + 4 - 1) linearly independent constraint equations. Obviously, we also require nonnegativity of the decision variables, since a negative shipment is impossible and does not have any valid interpretation."""

# ╔═╡ 5960fea9-007d-4151-9354-a1f215a8987e
let
	model = Model(GLPK.Optimizer)
	@variable(model, 0 <= x[1:3,1:4])
	@objective(model, Min, 7x[1,1]+10x[1,2]+14x[1,3]+8x[1,4]+7x[2,1]+11x[2,2]+12x[2,3]+6x[2,4]+5x[3,1]+8x[3,2]+15x[3,3]+9x[3,4])
	@constraint(model, con1, sum(x[1,j] for j in 1:4) == 30)
	@constraint(model, con2, sum(x[2,j] for j in 1:4) == 40)
	@constraint(model, con3, sum(x[3,j] for j in 1:4) == 30)
	@constraint(model, con4, sum(x[i,1] for i in 1:3) == 20)
	@constraint(model, con5, sum(x[i,2] for i in 1:3) == 20)
	@constraint(model, con6, sum(x[i,3] for i in 1:3) == 25)
	@constraint(model, con7, sum(x[i,4] for i in 1:3) == 35)
	optimize!(model)
	termination_status(model), primal_status(model), value.(x), objective_value(model)
end

# ╔═╡ 6e6423b6-52fa-43e0-b556-03c670ed9b7a
md"""This problem is an _integer linear programming_ problem, i.e. the solution components must be integers.

We can use the simplex method to find a solution to an ILP problem if the $m\times n$ matrix $A$ is unimodular, i.e. if all its nonzero $m$th order minors are $\pm 1$.

Should this not be the case, you can always impose a constraint on `x` to be integer. In doing so, you will (unkowingly) select a different solver for the problem.
"""

# ╔═╡ c6854393-3671-4858-b1ec-92f77a291b6a
let
	model = Model(GLPK.Optimizer)
	# specify that we need integer values
	@variable(model, 0 <= x[1:3,1:4], Int)
	@objective(model, Min, 7x[1,1]+10x[1,2]+14x[1,3]+8x[1,4]+7x[2,1]+11x[2,2]+12x[2,3]+6x[2,4]+5x[3,1]+8x[3,2]+15x[3,3]+9x[3,4])
	@constraint(model, con1, sum(x[1,j] for j in 1:4) == 30)
	@constraint(model, con2, sum(x[2,j] for j in 1:4) == 40)
	@constraint(model, con3, sum(x[3,j] for j in 1:4) == 30)
	@constraint(model, con4, sum(x[i,1] for i in 1:3) == 20)
	@constraint(model, con5, sum(x[i,2] for i in 1:3) == 20)
	@constraint(model, con6, sum(x[i,3] for i in 1:3) == 25)
	@constraint(model, con7, sum(x[i,4] for i in 1:3) == 35)
	optimize!(model)
	termination_status(model), primal_status(model), value.(x), objective_value(model)
end

# ╔═╡ 21f11662-4da4-47d1-af0b-f82b167f9708
md"""#### Electricity

An electric circuit is designed to use a 30 V source to charge 10 V, 6 V, and 20 V batteries connected in parallel. Physical constraints limit the currents $I_1$, $I_2$, $I_3$, $I_4$, and $I_5$ to a maximum of 4 A, 3 A, 3 A, 2 A, and 2 A, respectively. In addition, the batteries must not be discharged, that is, the currents $I_1$, $I_2$, $I_3$, $I_4$, and $I_5$ must not be negative. We wish to find the values of the currents $I_1$, $I_2$, $I_3$, $I_4$, and $I_5$ such that the total power transferred to the batteries is maximized.

The total power transferred to the batteries is the sum of the powers transferred to each battery, and is given by $10I_2 + 6I_4 + 20I_5$ W. From the circuit, we observe that the currents satisfy the constraints $I_1 = I_2 + I_3$, and $I_3 = I_4 + I_5$. Therefore, the problem can be posed as the following linear program:

```math
\max 10I_2+6I_4+20I_5
```

subject to

```math
\begin{aligned}
I_1 &= I_2 + I_3\\
I_3 &= I_4 + I_5\\
I_1 &\le 4\\
I_2 &\le 3\\
I_3 &\le 3\\
I_4 &\le 2\\
I_5 &\le 2
\end{aligned}
```

"""

# ╔═╡ fe731655-3319-4009-b5d5-778af69d9198
let
	model = Model(GLPK.Optimizer)
	@variable(model, 0 <= I[1:5])
	@objective(model, Max, 10*I[2]+6*I[4]+20I[5])
	@constraint(model, con1, I[1] == I[2] + I[3])
	@constraint(model, con2, I[3] == I[4] + I[5])
	@constraint(model, con3, I[1] <= 4)
	@constraint(model, con4, I[2] <= 3)
	@constraint(model, con5, I[3] <= 3)
	@constraint(model, con6, I[4] <= 2)
	@constraint(model, con7, I[5] <= 2)
	optimize!(model)
	termination_status(model), primal_status(model), value.(I), objective_value(model)
end

# ╔═╡ b2699452-d0fb-443a-bf6c-7d8c10a1717a
md"""#### Telecom

Consider a wireless communication system. There are $n$ "mobile" users. For each $i$ in $1,\dots, n$; user $i$ transmits a signal to the base station with power $P_i$ and an attenuation factor of $h_i$ (i.e., the actual received signal power at the basestation from user $i$ is $h_iP_i$). When the basestation is receiving from user $i$, the total received power from all other users is considered "interference" (i.e., the interference for user $i$ is $\sum_{i\ne j}h_jP_j$). For the communication with user $i$ to be reliable, the signal-to-interference ratio must exceed a threshold $\gamma_i$, where the "signal" is the received power for user $i$.

We are interested in minimizing the total power transmitted by all the users subject to having reliable communications for all users. We can formulate the problem as a linear programming problem of the form

```math
\min \sum_iP_i
```

subject to

```math
\forall i \in 1,\dots,n\,:\,\begin{cases}
\frac{h_iP_i}{\sum_{i\ne j}h_jP_j}\ge\gamma_i\\
P_i\ge0
\end{cases}
```"""

# ╔═╡ 1c05c58c-f73d-499f-81d9-c071bb931362
# Try it yourself

# ╔═╡ 282c0f80-015b-11eb-19d8-d9235a2a77d4
md"""## Global optimisation algorithms
We now broaden our scope and turn our attention to problems where the objective function is no longer linear. This section will cover unconstrained optimisation.
"""

# ╔═╡ 4d9273a0-015a-11eb-2caa-130d0245a946
md"""The solution of the unconstrained optimization problem

```math
\min f\left(\vec x\right)
```

can always be found by finding a critical point $\vec{x}^*$, such that (first order condition):

```math
\nabla f\left(\vec x^\star\right) = \vec 0\,.
```

This point can be a local minimum, maximum or a saddle point. If the Hessian, $\mathcal{H}f(\vec{x}^*$), is positive definite, we know it is a minimum local minimum (second order condition). Additionally, if the function is convex, we know that this minimum is a global minimum (cf. ES122 and ES221).

In practice, we choose an initial point and the generate a sequence of iterates. Typically, the best we can hope for is that the sequence converges to a local minimizer. For this reason, it is often desirable for the initial point to be close to a global optimizer. Most numerical require at least first derivatives or an approximation of it (and als second derivatives in the case of Newton's method), which can be limiting. We will discuss a method that is global in nature in the sense that they attempt to search throughout the entire feasible set. 

The _Nelder-Mead simplex_ method (detailed below) uses only objective function values and do not require derivatives. Consequently, it is applicable to a much wider class of optimization problems."""

# ╔═╡ bfe6caf0-015a-11eb-23a0-919e0520bd8d
md"""### The Nelder-Mead Simplex Algorithm

The Nelder-Mead algorithm uses the concept of a _simplex_. A simplex is a geometric object determined by an assembly of $n+1$ points, $\vec{p}_{0},\vec{p}_{1},\dots,\vec{p}_{n}$, in the n-dimensional space such that

```math
\det\begin{bmatrix}
\vec{p}_{0} & \vec{p}_{1} & \cdots & \vec{p}_{n}\\
1 & 1 & \cdots & 1
\end{bmatrix}\neq0\,.
```

This condition ensures that two points in $\mathbb R$ do not coincide, three points in $\mathbb R^{2}$ are not collinear, four points in $\mathbb R^{3}$ are not coplanar, and so on. Thus, a simplex in $\mathbb R$ is a line segment, in $\mathbb R^{2}$ it is a triangle, while a simplex in $\mathbb R^{3}$ is a tetrahedron; in each case it encloses a finite $n$-dimensional volume.

Suppose that we wish to minimize $f\left(\vec{x}\right),\,\vec{x}\in\mathbb R^{n}$.
To start the algorithm, we initialize a simplex of $n+1$ points.
A possible way to set up a simplex is to start with an initial point
$\vec{x}^{\left(0\right)}=\vec{p}_{0}$ and generate the remaining
points of the initial simplex as follows:

```math
\vec{p}_{i}=\vec{p}_{0}+\lambda_{i}\vec{e}_{i},\quad i=1,2,\dots,n\:,
```

where the $\vec{e}_{i}$ are unit vectors constituting the natural
basis of $\mathbb R^{n}$. The positive constant coefficients $\lambda_{i}$
are selected in such a way that their magnitudes reflect the length
scale of the optimization problem.

Our objective is to modify the initial simplex stage by stage so that
the resulting simplices converge toward the minimizer. At each iteration
we evaluate the function $f$ at each point of the simplex. In the
function minimization process, the point with the largest function
value is replaced with another point. The process for modifying the
simplex continues until it converges toward the function minimizer.

We now present the rules for modifying the simplex stage by stage.
To aid in our presentation, we use a two-dimensional example to illustrate
the rules. We begin by selecting the initial set of $n+1$ points
that are to form the initial simplex. We next evaluate $f$ at each
point and order the $n+1$ vertices to satisfy

```math
f\left(\vec{p}_{0}\right)\leq f\left(\vec{p}_{1}\right)\leq\cdots\leq f\left(\vec{p}_{n}\right)\,.
```

For the two-dimensional case we let $\vec{p}_{l}$, $\vec{p}_{n}$
and $\vec{p}_{s}$ denote the points of the simplex for which $f$
is largest, next largest and smallest. We next compute $\vec{p}_{g}$,
the _centroid_ (center of gravity) of the best $n$ points:

```math
\vec{p}_{g}=\frac{1}{n}\sum_{i=0}^{n-1}\vec{p}_{i}\,.
```

We then reflect the worst vertex, $\vec{p}_{n}$, in $\vec{p}_{g}$
using a _reflection coefficient_ $\rho>0$ to obtain the reflection
point

```math
\vec{p}_{r}=\vec{p}_{g}+\rho\left(\vec{p}_{g}-\vec{p}_{n}\right)\,.
```"""

# ╔═╡ 0aa85db0-015b-11eb-01ae-694c21b1f7a4
Drawing(width = "12cm", height = "6cm", viewBox = "0 0 600 300", preserveAspectRatio="none") do
    circle(cx = "52", cy = "150", r = "5", fill = "black")
    circle(cx = "250", cy = "50", r = "5", fill = "black")
    circle(cx = "250", cy = "250", r = "5", fill = "black")
    circle(cx = "250", cy = "150", r = "5", fill = "black")
    circle(cx = "450", cy = "150", r = "5", fill = "black")
    polyline(
        points="50, 150 250, 150 250, 50 50, 150 250, 250 250, 150 450 150", 
        fill="none", stroke="black", stroke_width="2.5"
    )
    NativeSVG.text(x = "40", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pl") end
    NativeSVG.text(x = "260", y = "40", width = "60", height = "60", font_size = "16pt") do
		str("pn") end
    NativeSVG.text(x = "260", y = "240", width = "60", height = "60", font_size = "16pt") do
		str("ps") end
    NativeSVG.text(x = "210", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pg") end
    NativeSVG.text(x = "440", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pr") end
end

# ╔═╡ c6569c40-016d-11eb-2d6b-4d8a54cff763
md"""A typical value is $\rho=1$. 

We proceed to evaluate $f$ at $\vec{p}_{r}$. If $f\left(\vec{p}_{0}\right)\leq f\left(\vec{p}_{r}\right)<f\left(\vec{p}_{n-1}\right)$,
then the point $\vec{p}_{r}$ replaces $\vec{p}_{l}$ to form a new
simplex and we terminate the iteration. We proceed to repeat the process.

If , however, $f\left(\vec{p}_{r}\right)<f\left(\vec{p}_{0}\right)$,
so that the point $\vec{p}_{r}$ yields the smallest function value
among the points of the simplex, we argue that this direction is a
good one. In this case we increase the distance traveled using an
_expansion coefficient_ $\chi>1$, e.g. $\chi=2$, to obtain

```math
\vec{p}_{e}=\vec{p}_{g}+\chi\left(\vec{p}_{r}-\vec{p}_{g}\right)\,.
```
"""

# ╔═╡ 0e6ae94e-016e-11eb-3f19-676f24836ebb
Drawing(width = "16cm", height = "6cm", viewBox = "0 0 800 300", preserveAspectRatio="none") do
    circle(cx = "50", cy = "150", r = "5", fill = "black")
    circle(cx = "250", cy = "50", r = "5", fill = "black")
    circle(cx = "250", cy = "250", r = "5", fill = "black")
    circle(cx = "250", cy = "150", r = "5", fill = "black")
    circle(cx = "450", cy = "150", r = "5", fill = "black")
    circle(cx = "650", cy = "150", r = "5", fill = "black")
    polyline(
        points="50, 150 250, 150 250, 50 50, 150 250, 250 250, 150 650 150", 
        fill="none", stroke="black", stroke_width="2.5"
    )
    NativeSVG.text(x = "40", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pl") end
    NativeSVG.text(x = "260", y = "40", width = "60", height = "60", font_size = "16pt") do
		str("pn") end
    NativeSVG.text(x = "260", y = "240", width = "60", height = "60", font_size = "16pt") do
		str("ps") end
    NativeSVG.text(x = "210", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pg") end
    NativeSVG.text(x = "440", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pr") end
    NativeSVG.text(x = "640", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pe") end
end

# ╔═╡ 6bf23d80-016e-11eb-1c64-6d2da5f0f954
md"""This operation yields a new point on the line $\vec{p}_{n}\vec{p}_{g}\vec{p}_{r}$
extend beyond $\vec{p}_{r}$. If $f\left(\vec{p}_{e}\right)<f\left(\vec{p}_{r}\right)$,
the expansion is declared a success and $\vec{p}_{e}$ replaces $\vec{p}_{n}$
in the next simplex. If, on the other hand, $f\left(\vec{p}_{e}\right)\geq f\left(\vec{p}_{r}\right)$,
the expansion is a failure and $\vec{p}_{r}$ replaces $\vec{p}_{n}$.

Finally, if $f\left(\vec{p}_{r}\right)\geq f\left(\vec{p}_{n-1}\right)$,
the reflected point $\vec{p}_{r}$ would constitute the point with
the largest function value in the new simplex. Then in the next step
it would be reflected in $\vec{p}_{g}$, probably an unfruitful operation.
Instead, this case is dealt with by a _contraction_ operation
in one of two ways. First, if $f\left(\vec{p}_{r}\right)<f\left(\vec{p}_{n}\right)$,
then we contract $\left(\vec{p}_{r}-\vec{p}_{g}\right)$ with a _contraction
coefficient_ $0<\gamma<1$, e.g. $\gamma=\frac{1}{2}$, to obtain"""

# ╔═╡ 7cf25a20-016e-11eb-30f1-8394316a43c2
Drawing(width = "16cm", height = "6cm", viewBox = "0 0 800 300", preserveAspectRatio="none") do
    circle(cx = "50", cy = "150", r = "5", fill = "black")
    circle(cx = "250", cy = "50", r = "5", fill = "black")
    circle(cx = "250", cy = "250", r = "5", fill = "black")
    circle(cx = "250", cy = "150", r = "5", fill = "black")
    circle(cx = "450", cy = "150", r = "5", fill = "black")
    circle(cx = "350", cy = "150", r = "5", fill = "black")
    polyline(
        points="50, 150 250, 150 250, 50 50, 150 250, 250 250, 150 450 150", 
        fill="none", stroke="black", stroke_width="2.5"
    )
    NativeSVG.text(x = "40", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pl") end
    NativeSVG.text(x = "260", y = "40", width = "60", height = "60", font_size = "16pt") do
		str("pn") end
    NativeSVG.text(x = "260", y = "240", width = "60", height = "60", font_size = "16pt") do
		str("ps") end
    NativeSVG.text(x = "210", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pg") end
    NativeSVG.text(x = "440", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pr") end
    NativeSVG.text(x = "340", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pc") end
end

# ╔═╡ c5157440-016e-11eb-33fd-4b9c155cf832
md"""```math
\vec{p}_{c}=\vec{p}_{g}+\gamma\left(\vec{p}_{r}-\vec{p}_{g}\right)\,.
```

We refer to this operation as the _outside contraction_. If,
on the other hand, $f\left(\vec{p}_{r}\right)\geq f\left(\vec{p}_{n}\right)$,
then $\vec{p}_{l}$ replaces $\vec{p}_{r}$ in the contraction operation
and we get

```math
\vec{p}_{c}=\vec{p}_{g}+\gamma\left(\vec{p}_{l}-\vec{p}_{g}\right)\,.
```

This operation is referred to as the _inside contraction_. """

# ╔═╡ fce70140-016e-11eb-224a-3d863b6817b6
Drawing(width = "16cm", height = "6cm", viewBox = "0 0 800 300", preserveAspectRatio="none") do
    circle(cx = "50", cy = "150", r = "5", fill = "black")
    circle(cx = "250", cy = "50", r = "5", fill = "black")
    circle(cx = "250", cy = "250", r = "5", fill = "black")
    circle(cx = "250", cy = "150", r = "5", fill = "black")
    circle(cx = "150", cy = "150", r = "5", fill = "black")
    polyline(
        points="50, 150 250, 150 250, 50 50, 150 250, 250 250, 150", 
        fill="none", stroke="black", stroke_width="2.5"
    )
    NativeSVG.text(x = "40", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pl") end
    NativeSVG.text(x = "260", y = "40", width = "60", height = "60", font_size = "16pt") do
		str("pn") end
    NativeSVG.text(x = "260", y = "240", width = "60", height = "60", font_size = "16pt") do
		str("ps") end
    NativeSVG.text(x = "210", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pg") end
    NativeSVG.text(x = "140", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pc") end
end

# ╔═╡ 18554920-0186-11eb-0cb2-f5d020aff0d9
md"""If , in either case, $f\left(\vec{p}_{c}\right)\leq f\left(\vec{p}_{n}\right)$,
the contraction is considered a success, and we replace $\vec{p}_{n}$
with $\vec{p}_{c}$ in the new simplex. If, however, $f\left(\vec{p}_{c}\right)>f\left(\vec{p}_{n}\right)$,
the contraction is a failure, and in this case a new simplex can be
formed by retaining $\vec{p}_{0}$ only and halving the distance from
$\vec{p}_{0}$ to every point in the simplex. We can refer to this
event as a _shrinkage operation_. In general, the shrink step
produces the $n$ new vertices of the new simplex according to the
formula

```math
\vec{v}_{i}=\vec{p}_{0}+\sigma\left(\vec{p}_{i}-\vec{p}_{0}\right)\,,\quad i=1,2,\dots,n\,,
```"""

# ╔═╡ 392084d2-0186-11eb-11e5-2b5128129e19
Drawing(width = "16cm", height = "6cm", viewBox = "0 0 800 300", preserveAspectRatio="none") do
    circle(cx = "50", cy = "150", r = "5", fill = "black")
    circle(cx = "250", cy = "50", r = "5", fill = "black")
    circle(cx = "250", cy = "250", r = "5", fill = "black")
    circle(cx = "250", cy = "150", r = "5", fill = "black")
    circle(cx = "150", cy = "200", r = "5", fill = "black")
    polyline(
        points="250, 150 250, 50 50, 150 250, 250 250, 150 150, 200", 
        fill="none", stroke="black", stroke_width="2.5"
    )
	NativeSVG.text(x = "40", y = "170", width = "60", height = "60", font_size = "16pt") do
		str("pl") end
    NativeSVG.text(x = "260", y = "40", width = "60", height = "60", font_size = "16pt") do
		str("pn") end
    NativeSVG.text(x = "260", y = "240", width = "60", height = "60", font_size = "16pt") do
		str("ps") end
end

# ╔═╡ 6e95360e-0186-11eb-1ed3-770285bc0e74
md"""where $\sigma=\frac{1}{2}$. Hence, the vertices of the new simplex
are $\vec{p}_{0},\vec{v}_{1},\dots,\vec{v}_{n}$.

When implementing the simplex algorithm, we need a tie-breaking rule
to order points in the case of equal function values. Often, the highest
possible index consistent with the relation

```math
f\left(\vec{p}_{0}\right)\leq f\left(\vec{p}_{1}\right)\leq\cdots\leq f\left(\vec{p}_{n}\right)
```

is assigned to the new vertex."""

# ╔═╡ 86bb011e-0186-11eb-2b14-8ddd7f9ca6c2
md"""### Implementations
Within Julia, the [`Optim`](https://github.com/JuliaNLSolvers/Optim.jl) package contains a lot of different optimisation routines.

To illustrate the packaga's functionality, we minimize the Rosenbrock function, a classical test problem for numerical optimization, using different optimisation methods."""

# ╔═╡ 202572f0-0187-11eb-0172-9bea075d89a9
# function definition
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2;

# ╔═╡ d332f672-0186-11eb-205a-97aeef39b75d
begin
	x, y, z = let
		x = -1:0.007:2
		y = -1:0.007:2
		z = Surface((x,y)->log10.(f([x,y])), x, y)
		x, y, z
	end;
	contour(x,y,z, xlabel=L"x_1", ylabel=L"x_2", title="Rosenbrock function")
end

# ╔═╡ 4680a1e0-0187-11eb-3a9c-5b631ea7c822
md"""Once we've defined this function, we can find the minimizer (the input that minimizes the objective) and the minimum (the value of the objective at the minimizer) using any of our favorite optimization algorithms. With a function defined, we just specify an initial point x and call optimize with a starting point `x0`:"""

# ╔═╡ 5815d290-0187-11eb-391d-ff073eb9dc85
x0 = [0.0, 0.0]; # initial point

# ╔═╡ 0589ddc0-0c2d-42dc-b526-6334144b4476
md"#### Nelder-Mead"

# ╔═╡ 75d97390-0187-11eb-1b3f-69c879780178
optimize(f, x0, NelderMead())

# ╔═╡ 254bd98d-db2c-4eb8-9bb9-85e574420254
md"#### Quasi-Newton methods"

# ╔═╡ 7927b430-0187-11eb-345f-a10499db73fa
md"""Other local solvers are available. Below, we use BFGS, a quasi-Newton method that requires a gradient. If we only provide the function $f$, Optim will construct an approximate gradient for us using central finite differencing (cf. ES221):"""

# ╔═╡ 86f5f1d0-0187-11eb-1186-ad0df4a1ef6c
optimize(f, x0, BFGS())

# ╔═╡ 8dd8f1f0-0187-11eb-370a-a1b5053e91e4
md"""For better performance and greater precision, you can pass your own (analytical) gradient function. If your objective is written in all Julia code with no special calls to external (that is non-Julia) libraries, you can also use [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), by using the `autodiff` keyword and setting it to `:forward` (to us forward mode):"""

# ╔═╡ 97810580-0187-11eb-38a3-fd83c085e89c
optimize(f, x0, BFGS(); autodiff=:forward)

# ╔═╡ 9fb64f30-0187-11eb-1a26-7f9e577886d9
md"""For the Rosenbrock example, the analytical gradient can be shown to be:"""

# ╔═╡ ad0c2880-0187-11eb-31c1-bdfd311b2650
function g!(G, x)
	G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
	G[2] = 200.0 * (x[2] - x[1]^2)
end

# ╔═╡ ba3db0f0-0187-11eb-2efa-296cff31defa
md"""Note, that the functions we're using to calculate the gradient (and later the Hessian `h!`) of the Rosenbrock function mutate a fixed-sized storage array, which is passed as an additional argument called `G` (or `H` for the Hessian) in these examples. By mutating a single array over many iterations, this style of function definition removes the sometimes considerable costs associated with allocating a new array during each call to the `g!` or `h!` functions. """

# ╔═╡ c818e6e0-0187-11eb-2c4a-b78c74a10231
optimize(f, g!, x0, BFGS())

# ╔═╡ 0e4f381b-e6bc-4be3-a18b-d22aa1a5fc73
md"#### Newton's method"

# ╔═╡ cd8db100-0187-11eb-0ac7-2f4510dc2f08
md"""In addition to providing gradients, you can provide a Hessian function `h!` as well. In our current case this is"""

# ╔═╡ d80597b0-0187-11eb-23ef-7daa3ac30f36
function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

# ╔═╡ e07a5b60-0187-11eb-246d-f18cbf91b1f1
md"""Now we can use Newton's method for optimization by running:"""

# ╔═╡ f2674900-0187-11eb-017d-e5d26651604d
optimize(f, g!, h!, x0)

# ╔═╡ f933b3e0-0187-11eb-113f-876918888691
md"""Which defaults to `Newton()` since a Hessian function was provided. Like gradients, the Hessian function will be ignored if you use a method that does not require it.

Note that `Optim` will not generate approximate Hessians using finite differencing because of the potentially low accuracy of approximations to the Hessians. Other than Newton's method, none of the algorithms provided by the Optim package employ exact Hessians."""

# ╔═╡ 2837d5e2-0188-11eb-1eb3-67c8aeb366d0
md"""### Choosing an appropriate algorithm

The first choice to be made is that of the order of the method. Zeroth-order methods do not have gradient information, and are very slow to converge, especially in high dimension. First-order methods do not have access to curvature information and can take a large number of iterations to converge for badly conditioned problems. Second-order methods can converge very quickly once in the vicinity of a minimizer. Of course, this enhanced performance comes at a cost: the objective function has to be differentiable, you have to supply gradients and Hessians, and, for second order methods, a linear system has to be solved at each step.

If you can provide analytic gradients and Hessians, and the dimension of the problem is not too large, then second order methods are very efficient. The _Newton method with trust region_ is the method of choice.

When you do not have an explicit Hessian or when the dimension becomes large enough that the linear solve in the Newton method becomes the bottleneck, first order methods should be preferred. _BFGS_ is a very efficient method, but also requires a linear system solve. _L-BFGS_ usually has a performance very close to that of BFGS, and avoids linear system solves (the parameter `m` can be tweaked: increasing it can improve the convergence, at the expense of memory and time spent in linear algebra operations). The _conjugate gradient method_ usually converges less quickly than L-BFGS, but requires less memory. _Gradient descent_ should only be used for testing.

When the objective function is non-differentiable or you do not want to use gradients, use zeroth-order methods. _Nelder-Mead_ is currently the most robust."""

# ╔═╡ 35b66ce0-0188-11eb-2b93-fb6f4e8fe9ec
md"""### Linesearches

Linesearches are used in every first- and second-order method except for the trust-region Newton method. Linesearch routines attempt to locate quickly an approximate minimizer of the univariate function

```math
\alpha \rightarrow f\left(\vec x + \alpha \vec d\right)\,,
```

where $\vec d$ is the descent direction computed by the algorithm. They vary in how accurate this minimization is. Two good linesearches are BackTracking and HagerZhang, the former being less stringent than the latter. For well-conditioned objective functions and methods where the step is usually well-scaled (such as L-BFGS or Newton), a rough linesearch such as BackTracking is usually the most performant. For badly behaved problems or when extreme accuracy is needed (gradients below the square root of the machine epsilon, about `10−8` with Float64), the HagerZhang method proves more robust. An exception is the conjugate gradient method which requires an accurate linesearch to be efficient, and should be used with the HagerZhang linesearch."""

# ╔═╡ 648f6030-0188-11eb-2ca2-7b2e080de91b
md"""### Summary

As a very crude heuristic:

For a low-dimensional problem with analytic gradients and Hessians, use the Newton method with trust region. For larger problems or when there is no analytic Hessian, use L-BFGS, and tweak the parameter `m` if needed. If the function is non-differentiable, use Nelder-Mead.

Use the `HagerZhang` linesearch for robustness and `BackTracking` for speed.

Requires only a function handle:

- `NelderMead()`
- `SimulatedAnnealing()`, see next year
- `ParticleSwarm()`, see next year

Requires a function and gradient (will be approximated if omitted):

- `BFGS()`
- `LBFGS()`
- `ConjugateGradient()`
- `GradientDescent()`

Requires a function, a gradient, and a Hessian (cannot be omitted):

- `Newton()`
- `NewtonTrustRegion()`

Special methods for bounded univariate optimization:

- `Brent()`
- `GoldenSection()`"""

# ╔═╡ 3c2d2158-d160-49ec-9030-bf39b7ec75d4
md"""
## Global contrained optimisation algorithms
We can now turn our attention to global optimisation algorithms including constraints. We will start out with a specific case, namely quadratic programming problems.

### Quadratic Programming

An optimization problem with a quadratic objective function and linear
constraints is called a _quadratic program_. Problems of this
type are important in their own right, and they also arise a subproblems
in methods for general constrained optimization such as sequential
quadratic programming and interior-point methods.

The general quadratic program can be stated as

```math
\begin{aligned}
\min_{\vec{x}}\, & f\left(\vec{x}\right)\overset{\vartriangle}{=}\frac{1}{2}\vec{x}^\mathsf{T}Q\vec{x}-\vec{c}^\mathsf{T}\vec{x}\\
\textrm{subject to}\, & \begin{cases}
A_{\textrm{eq}}\vec{x}=\vec{b}_{\textrm{eq}}\,,\\
A_{\textrm{in}}\vec{x}\leq\vec{b}_{\textrm{in}}\,,
\end{cases}
\end{aligned}
```

where $Q$ is a symmetric $n\times n$ matrix, $\vec{c}\in\mathbb R^{n}$,
$A_{\textrm{eq}}$ is a $m\times n$ matrix, $\vec{b}_{\textrm{eq}}\in\mathbb R^{m}$,
$A_{\textrm{in}}$ is a $p\times n$ matrix and $\vec{b}_{\textrm{in}}\in\mathbb R^{p}$.
If the Hessian matrix $Q$ is positive semidefinite, we have a convex
quadratic program. Non-convex quadratic programs, in which $Q$ is
an indefinite matrix, can be more challenging because they have several
stationary points and local minima.


#### Equality Constraints

We begin our discussion of algorithms for quadratic programming by considering
the case in which only equality constraints are present. We consider
to following equality-constrained quadratic problem
```math
\begin{aligned}
\min_{\vec{x}}\, & f\left(\vec{x}\right)\overset{\vartriangle}{=}\frac{1}{2} \vec{x}^\mathsf{T}Q\vec{x}- \vec{c}^\mathsf{T}\vec{x}\\
\textrm{subject to}\, & A\vec{x}=\vec{b}\,,\nonumber 
\end{aligned}
```
where $A$ is the $m\times n$ Jacobian of constraints and $\vec{b}\in\mathbb R^{m}$.
We assume that $A$ has rank $m$ so that the constraints are consistent.

The First Order Necessary Condition (FONC) uses the Langrangian function
```math
\mathcal L\left(\vec x, \vec \lambda\right)=\vec{x}^\mathsf{T}Q\vec{x}- \vec{c}^\mathsf{T}\vec{x} + \vec\lambda^\mathsf{T} \left(A\vec{x}-\vec{b}\right)
```
and gives the following condition
```math
\begin{aligned}
\nabla_{\vec x}\mathcal L\left(\vec x ^\star, \vec \lambda ^\star\right)&=\vec 0\,,\\
\nabla_{\vec \lambda}\mathcal L\left(\vec x ^\star, \vec \lambda ^\star\right)&=\vec 0\,,
\end{aligned}
```
So the FONC for $\vec{x}^{\star}$ to be a solution of the quadratic
problem yields a vector $\vec{\lambda}^{\star}$ such
that the following system of equations is satisfied:
```math
\begin{pmatrix}Q & A^\mathsf{T}\\
A & 0
\end{pmatrix}\begin{pmatrix}\vec{x}^{\star}\\
\vec{\lambda}^{\star}
\end{pmatrix}=\begin{pmatrix}\vec{c}\\
\vec{b}
\end{pmatrix}\,.
```
This system can be solved directly by factorization. An alternative
is to use an iterative method.
"""

# ╔═╡ ad19fcc2-c149-40d6-b2eb-c112312f066a
md"""
#### Example
Consider the quadratic programming problem

```math
\begin{aligned}
\min_{\vec{x}}\, & \frac{1}{2} \vec{x}^\mathsf{T}\begin{pmatrix}6 & 2 & 1\\
2 & 5 & 2\\
1 & 2 & 4
\end{pmatrix}\vec{x}- \begin{pmatrix}8\\
3\\
3
\end{pmatrix}^\mathsf{T}\vec{x}\\
\textrm{subject to}\, & \begin{pmatrix}1 & 0 & 1\\
0 & 1 & 1
\end{pmatrix}\vec{x}=\begin{pmatrix}3\\
0
\end{pmatrix}\,.
\end{aligned}
```
"""

# ╔═╡ a0b5b8b0-14e0-4a50-abec-74f9130feed2
let
	Q = [6 2 1
     	 2 5 2
         1 2 4]
	c = [8, 3, 3]
	A = [1 0 1
		 0 1 1]
	b = [3, 0]
	sol = [Q transpose(A)
		   A zeros(2,2)] \ [c; b]
	@info """solution 
	 - x: $(sol[1:3]), 
	 - λ: $(sol[4:5])
	 - f: $(1/2 * sol[1:3]' * Q * sol[1:3] - c' * sol[1:3] )"""
end

# ╔═╡ 1af9b16a-b9dd-4bcb-bc9d-035b7cf18568
md"""
#### General Constraints
##### Active Set Method

We now describe active-set methods for solving quadratic programs
containing both equality and inequality constraints. We consider only
the convex case, in which the matrix ``Q`` is positive semidefinite.

If the contents of the optimal active set ``J\left(\vec x^\star\right)`` were known in advance, we could find the solution ``\vec{x}^{\star}``
by applying the technique for equality constrained quadratic programs
to the problem
```math
\begin{aligned}
\min_{\vec{x}}\, & f\left(\vec{x}\right)\overset{\vartriangle}{=}\frac{1}{2} \vec{x}^\mathsf{T}Q\vec{x}- \vec{c}^\mathsf{T}\vec{x}\\
\textrm{subject to}\, & \begin{cases}
A_{\textrm{eq}}\vec{x}=\vec{b}_{\textrm{eq}}\,,\\
 \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{x}=b_{\textrm{in},j}\,, & \forall j\in J\left(\vec{x}^{\star}\right)\,,
\end{cases}
\end{aligned}
```
where ``\vec{a}_{\textrm{in},j}`` is the ``j``th row in the matrix ``A_{\textrm{in}}``
and ``b_{\textrm{in},j}`` is the ``j``th element of the vector ``\vec{b}_{\textrm{in}}``.
Of course, we usually do not have prior knowledge of ``J\left(\vec{x}^{\star}\right)``
and determination of this set is the main challenge facing algorithms
for inequality-constrained quadratic programs.

Active-set methods find a step from one iterate to the next by solving
a quadratic subproblem in which some of the inequality constraints,
and all the equality constraints are imposed as equalities. This subset
is referred to as the _working set_ and is denoted at the ``k``th
iterate by ``W_{k}``. An important requirement we impose on ``W_{k}``
is that the gradients ``\vec{a}_{\textrm{eq},i}``, ``i=1,\dots,m`` and
``\vec{a}_{\textrm{in},j}``, ``j\in W_{k}`` are linearly independent,
even when the full set of active constraints at that point has linearly
dependent gradients.

Given an iterate ``\vec{x}^{\left(k\right)}`` and the working set ``W_{k}``,
we first check whether ``\vec{x}^{\left(k\right)}`` minimizes the quadratic
function ``f`` in the subspace defined by the working set. If not,
we compute a step ``\vec{d}^{\left(k\right)}`` by solving an equality-constrained
quadratic subproblem in which the inequality constraints corresponding
to the working set ``W_{k}`` are regarded as equalities and the other
inequality constraints are temporarily disregarded. To express this
subproblem in terms of the step ``\vec{d}^{\left(k\right)}``, we define
```math
\vec{d}^{\left(k\right)}=\vec{x}^{\left(k+1\right)}-\vec{x}^{\left(k\right)}\,,\quad\vec{g}_{k}=Q\vec{x}^{\left(k\right)}-\vec{c}\,.
```
By substituting for ``\vec{x}^{\left(k+1\right)}`` into the objective
function, we find that
```math
f\left(\vec{x}^{\left(k+1\right)}\right)=f\left(\vec{x}^{\left(k\right)}+\vec{d}^{\left(k\right)}\right)=\frac{1}{2}\left(\vec{d}^{\left(k\right)}\right)^\mathsf{T}Q\vec{d}^{\left(k\right)}+\vec{g}_{k}^\mathsf{T}\vec{d}^{\left(k\right)}+\rho_{k}\,,
```
where ``\rho_{k}=\frac{1}{2} \left(\vec{x}^{\left(k\right)}\right)^\mathsf{T}Q\vec{x}^{\left(k\right)}- \vec{c}^\mathsf{T}\vec{x}^{\left(k\right)}``
is independent of ``\vec{d}^{\left(k\right)}``. Since we can drop ``\rho_{k}``
from the objective function without changing the solution of the problem,
we can write the subproblem to be solved at the ``k``th iteration as
follows
```math
    \min_{\vec{d}^{\left(k\right)}} \frac{1}{2} \left(\vec{d}^{\left(k\right)}\right)^\mathsf{T}Q\vec{d}^{\left(k\right)}+ \vec{g}_{k}^\mathsf{T}\vec{d}^{\left(k\right)} \text{ subject to}
    \begin{cases}
        A_{\text{eq}}\vec{d}^{\left(k\right)}=\vec{0} \\
        \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{d}^{\left(k\right)}=0
    \end{cases}
    , \forall j\in W_{k}
```


Note that for each ``j\in W_{k}``, the value of `` \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{x}^{\left(k\right)}``
does not change as we move along ``\vec{d}^{\left(k\right)}``, since
we have `` \vec{a}_{\textrm{in},j}^\mathsf{T}\left(\vec{x}^{\left(k\right)}+\alpha\vec{d}^{\left(k\right)}\right)= \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{x}^{\left(k\right)}=b_{\textrm{in},j}``
for all ``\alpha``. Since the constraints in ``W_{k}`` were satisfied
at ``\vec{x}^{\left(k\right)}``, they are also satisfied at ``\vec{x}^{\left(k\right)}+\alpha\vec{d}^{\left(k\right)}``,
for any value of ``\alpha``.

Supposing for the moment that the optimal ``\vec{d}^{\left(k\right)}``
is nonzero, we need to decide how far to move along this direction.
If ``\vec{x}^{\left(k\right)}+\vec{d}^{\left(k\right)}`` is feasible
with respect to all the constraints, we set ``\vec{x}^{\left(k+1\right)}=\vec{x}^{\left(k\right)}+\vec{d}^{\left(k\right)}``.
Otherwise, we set
```math
\vec{x}^{\left(k+1\right)}=\vec{x}^{\left(k\right)}+\alpha_{k}\vec{d}^{\left(k\right)}\,,
```
where the step-length parameter ``\alpha_{k}`` is chosen to be the
largest value in the range ``\left[0,1\right]`` for which all constraints
are satisfied. We can derive an explicit definition of ``\alpha_{k}``
by considering what happens to the constraints ``j\notin W_{k}``, since
the constraints ``j\in W_{k}`` will certainly be satisfied regardless
of the choice of ``\alpha_{k}``. If `` \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{d}^{\left(k\right)}\leq0``
for some ``j\notin W_{k}``, then for all ``\alpha_{k}\geq0``, we have
`` \vec{a}_{\textrm{in},j}^\mathsf{T}\left(\vec{x}^{\left(k\right)}+\alpha_{k}\vec{d}^{\left(k\right)}\right)\leq \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{x}^{\left(k\right)}\leq b_{\textrm{in},j}``.
Hence, constraint ``j`` will be satisfied for all nonnegative choices
of the step-length parameter. Whenever `` \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{d}^{\left(k\right)}>0``
for some ``j\notin W_{k}``, however, we have that `` \vec{a}_{\textrm{in},j}^\mathsf{T}\left(\vec{x}^{\left(k\right)}+\alpha_{k}\vec{d}^{\left(k\right)}\right)\leq b_{\textrm{in},j}``
only if
```math
\alpha_{k}\leq\frac{b_{\textrm{in},j}- \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{x}^{\left(k\right)}}{ \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{d}^{\left(k\right)}}\,.
```
To maximize the decrease in ``f``, we want ``\alpha_{k}`` to be as large
as possible in ``\left[0,1\right]`` subject to retaining feasibility,
so we obtain the following definition
```math
\alpha_{k}\overset{\textrm{def}}{=}\min\left\{ 1,\min_{j\notin W_{k}, \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{d}^{\left(k\right)}>0}\frac{b_{\textrm{in},j}- \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{x}^{\left(k\right)}}{ \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{d}^{\left(k\right)}}\right\} \,.
```
We call the constraints ``j`` for which this minimum is achieved the
_blocking constraints_. Note that it is quite possible for ``\alpha_{k}``
to be zero, because we could have `` \vec{a}_{\textrm{in},j}^\mathsf{T}\vec{d}^{\left(k\right)}>0``
for some constraint ``j`` that is active at ``\vec{x}^{\left(k\right)}``
but not a member of the current working set ``W_{k}``.

If ``\alpha_{k}<1``, that is, the step along ``\vec{d}_{k}`` was blocked
by some constraint not in ``W_{k}``, a new working set ``W_{k+1}`` is
constructed by adding one of the blocking constraints to ``W_{k}``.
We continue to iterate in this manner, adding constraints to the working
set until the subproblem has solution ``\vec{d}^{\circ}=\vec{0}``.
Since ``\vec{d}^{\circ}=\vec{0}`` satisfy the optimality condition, we have that
```math
\sum_{i=1}^{m}\lambda_{i}^{\circ}\vec{a}_{\textrm{eq},i}+\sum_{j\in W^{\circ}}\mu_{j}^{\circ}\vec{a}_{\textrm{in},j}=-\vec{g}^{\circ}=-Q\vec{x}^{\circ}+\vec{c}\,,
```
for some Lagrange multipliers ``\lambda_{i}^{\circ}``, ``i=1,\dots,m``
and ``\mu_{j}^{\circ}``, ``j\in W^{\circ}``. It follows that ``\vec{x}^{\circ}``,
``\vec{\lambda}^{\circ}`` and ``\vec{\mu}^{\circ}`` satisfy the second
KKT condition, if we define the multipliers corresponding to the inequality
constraints not in the working set to be zero. Because of the control
imposed on the step length, ``\vec{x}^{\circ}`` is also feasible with
respect to all the constraints, so the third, fourth and fifth KKT
conditions are satisfied at this point.

We now examine the signs of the KKT multipliers in the working set,
that is, the indices ``j\in W^{\circ}``. If these conditions are all
nonnegative, the first KKT condition is also satisfied, so we conclude
that ``\vec{x}^{\circ}`` is a KKT point for the original problem. In
fact, since ``Q`` is positive semidefinite, we have that ``\vec{x}^{\circ}``
is a global minimum.

If, on the other hand, on or more of the multipliers ``\mu_{j}^{\circ}``,
``j\in W^{\circ}``, are negative, the first KKT condition is not satisfied
and the objective function ``f`` may be decreased by dropping one of
these constraints. Thus, we remove an index ``j`` corresponding to
one of the negative multipliers from the working set and solve a new
subproblem for the next step. While any index ``j`` for which ``\mu_{j}^{\circ}<0``
usually will yield in a direction ``\vec{d}`` along which the algorithm
can make progress, the most negative multiplier is often chosen in
practice. This choice is motived by a sensitivity analysis, which
shows that the rate of decrease in the objective function when one
constraint is removed, is proportional to the magnitude of the Lagrange
multiplier for that constraint.
"""

# ╔═╡ 9256f823-1ea5-4dc7-86e4-7cd85c07433d
md"""
###### Example
Apply the active-set method to the following problem:
```math
\begin{aligned}
\min_{\vec{x}}\, & f\left(\vec{x}\right)=\left(x_{1}-1\right)^{2}+\left(x_{2}-2.5\right)^{2}\\
\textrm{subject to}\, & \begin{cases}
-x_{1}+2x_{2}-2\leq0\,,\\
x_{1}+2x_{2}-6\leq0\,.\\
x_{1}-2x_{2}-2\leq0\,,\\
-x_{1}\leq0\,,\\
-x_{2}\leq0\,.
\end{cases}
\end{aligned}
```
"""

# ╔═╡ 9a1cf74e-ce12-45ce-9e54-b8eddb7d4d83
begin
	# identifying the problem's characteristic elements
	Q = Float64[2 0
                0 2]
	c = Float64[2, 5]
	A = Float64[-1  2
				 1  2
				 1 -2
				-1  0
				 0 -1]
	b = Float64[2, 6, 2, 0, 0]
	nothing
end

# ╔═╡ 714ca8fe-ab51-4ecf-a5a8-8193fc3d8a28
md"An illustration of the function values and the feasible domain is shown below:"

# ╔═╡ fba69647-7a8a-4c3f-aa81-b743d72e7a99

let
	x = -1:0.05:5
	y = -1:0.05:5
	z = Surface((x,y)->(0.5 .* [x y]*Q*[x;y] .- transpose(c)*[x;y] .+ 0.5 .* transpose(c)*Q*c)[1], x, y)
	contour(x, y, z, levels=35)
	plot!(x, (2 .+ x) ./ 2, linestyle=:dash, label=L"-x_1+2x_2-2≤ 0")
	plot!(x, (6 .- x) ./ 2, linestyle=:dash, label=L"x_1+2x_2-6≤ 0")
	plot!(x, (2 .- x) ./ -2, linestyle=:dash, label=L"x_1-2x_2-2≤ 0")
	plot!([0,2,4,2,0,0],[0,0,1,2,1,0], linewidth=2, label="feasible domain", xlabel=L"x_1", ylabel=L"x_2", title=L"f : (x_1 - 1)^2 + (x_2 - 2.5)^2")
end

# ╔═╡ a26bac7f-d605-4285-b4bc-40b5e7da8745
md"""We refer the constraints, in order, by indices $1$ through $5$.

For this problem it is easy to determine a feasible initial point;
say $\vec{x}^{\left(0\right)}=\begin{pmatrix}2 & 0\end{pmatrix}\mathsf{T}$.

Constraints $3$ and $5$ are active at this point, and we set $W_{0}=\left\{ 3,5\right\} $.
Note that we could just as validly have chosen $W_{0}=\left\{ 5\right\} $
or $W_{0}=\left\{ 3\right\} $ or even $W_{0}=\emptyset$. Each choice
would lead the algorithm to perform somewhat differently."""

# ╔═╡ 0307b650-1d50-4348-9272-d5575d5308d4
x₀ = [2, 0]

# ╔═╡ 9259ba81-2026-4781-b438-e356b5cd4e4e
let x = x₀
	g = Q*x - c
	ain = [reshape(A[3,:], 1, 2); reshape(A[5,:], 1, 2)]
	sol = [Q transpose(ain)
		   ain zeros(2,2)] \ [-g; 0; 0]
end

# ╔═╡ c636a610-9157-42c0-8bda-767112b4b475
md"""Since $\vec{x}^{\left(0\right)}$ lies on a vertex of the feasible
region, it is obviously a minimizer of the objective function $f$
with respect to the working set $W_{0}$; that is, the solution of
the subproblem with $k=0$ is $\vec{d}^{\left(0\right)}=\vec{0}$.
We can then find the multipliers $\mu_{3}^{\circ}$ and $\mu_{5}^{\circ}$
associated with the active constraints. Substitution of the data from
our problem yields
```math
\begin{pmatrix}-1\\
2
\end{pmatrix}\lambda_{3}^{\circ}+\begin{pmatrix}0\\
1
\end{pmatrix}\lambda_{5}^{\circ}=\begin{pmatrix}2\\
-5
\end{pmatrix}\,,
```
which has solution $\lambda_{3}^{\circ}=-2$ and $\lambda_{5}^{\circ}=-1$. 

We now remove constraint $3$ from the working set, because it has
the most negative multiplier, and set $W_{1}=\{5\}$."""

# ╔═╡ 781240f7-0323-4be3-960e-6960902f3aaa
sol₁ = let x = x₀
	g = Q*x - c
	ain = reshape(A[5,:], 1, 2)
	sol = [Q transpose(ain)
		   ain zeros(1,1)] \ [-g; 0]
end

# ╔═╡ 7f09eb59-4c95-40a5-9858-1668c3c187b4
md"""We begin iteration $1$ by finding the solution of the subproblem for $k=1$, which is
$\vec{d}^{\left(1\right)}= \begin{pmatrix}-1 & 0\end{pmatrix}^\mathsf{T}$."""

# ╔═╡ a8ae98e2-59d1-45d3-a824-6072203465d9
α₁, d₁ = let x = x₀
	d = sol₁[1:2]
	α = min(1.0, [(reshape(A[j,:], 1, 2) * d)[1,1] ≤ 0 ? 1.0 : ((b[j] .- reshape(A[j,:], 1, 2) * x) / (reshape(A[j,:], 1, 2) * d))[1,1] for j in (1,2,3,4)]...)
	α, d
end

# ╔═╡ bd8a4b3f-86e5-4b32-847b-a2cf0ed6af04
md"""The step-length formula yields $\alpha_{1}=1$, and the new iterate
is $\vec{x}^{\left(2\right)}=\begin{pmatrix}1 & 0\end{pmatrix}^\mathsf{T}$. There are no blocking constraints, so that ``W_{2}=W_{1}=\left\{ 5\right\}``"""

# ╔═╡ 61780b00-1d06-4e8a-af35-3e0340d9fb5b
x₁ = x₀ + α₁ .* d₁

# ╔═╡ fded31f0-d136-4c6f-949b-40b2027b45ad
sol₂ = let x = x₁
	g = Q*x - c
	ain = reshape(A[5,:], 1, 2)
	sol = [Q transpose(ain)
		   ain zeros(1,1)] \ [-g; 0]
end

# ╔═╡ aeec3f14-f31f-455c-bd5c-1491b692b161
md"""We find at the start of iteration $2$ that the solution of the
subproblem is $\vec{d}^{\left(2\right)}=\vec{0}$. We deduce that
the Lagrange multiplier for the lone working constraint is $\lambda_{5}^{\circ}=-5$,
so we drop the working set to obtain $W_{3}=\emptyset$."""

# ╔═╡ 85d2c8cf-418d-4709-b3bc-df2dce39a1d2
x₂ = x₁

# ╔═╡ 9308ce1a-0fa5-4550-a203-48d8ea66f5ad
sol₃ = let x = x₂
	g = Q*x - c
	sol = Q \ (-g)
end

# ╔═╡ 591f1052-d1e5-4078-b010-eb78d69b7789
md"""Iteration $3$ starts by solving the unconstrained problem, to obtain
the solution $\vec{d}^{\left(3\right)}=\begin{pmatrix}0 & 2.5\end{pmatrix}^\mathsf{T}$."""

# ╔═╡ 6224ef69-0ecc-4bb1-8295-53335a719681
α₃, d₃ = let x = x₂
	d = sol₃[1:2]
	α = min(1.0, [(reshape(A[j,:], 1, 2) * d)[1,1] ≤ 0 ? 1.0 : ((b[j] .- reshape(A[j,:], 1, 2) * x) / (reshape(A[j,:], 1, 2) * d))[1,1] for j in (1,2,3,4,5)]...)
	α, d
end

# ╔═╡ e5cd3218-34d4-4b8d-b8ad-428110409acd
md"""The step-length formula yields a step length of $\alpha_{3}=0.6$
and a new iterate $\vec{x}^{\left(4\right)}=\begin{pmatrix}1 & 1.5\end{pmatrix}^\mathsf{T}$. There
is a single blocking constraint (constraint $1$), so we obtain ``W_{4}=\left\{ 1\right\}``."""

# ╔═╡ d57a4395-d2ea-4480-bbad-127e082378e6
x₃ = x₂ + α₃ .* d₃

# ╔═╡ 4f35e84d-871b-4418-b079-b61dcd05bcce
sol₄ = let x=x₃
	g = Q*x - c
	ain = reshape(A[1,:], 1, 2)
	sol = [Q transpose(ain)
		   ain zeros(1,1)] \ [-g; 0]
end

# ╔═╡ 77d30dfa-3538-4111-ae3d-b506bea0d81f
md"""The solution of the subproblem for $k=4$ is then $\vec{d}^{\left(4\right)}=\begin{pmatrix}0.4 & 0.2\end{pmatrix}^\mathsf{T}$."""

# ╔═╡ e3eec868-7191-4296-8aaa-bd61670b00da
α₄, d₄ = let x = x₃
	d = sol₄[1:2]
	α = min(1.0, [(reshape(A[j,:], 1, 2) * d)[1,1] ≤ 0 ? 1.0 : ((b[j] .- reshape(A[j,:], 1, 2) * x) / (reshape(A[j,:], 1, 2) * d))[1,1] for j in (2,3,4,5)]...)
	α, d
end

# ╔═╡ 612c70a9-c220-44ad-aff6-569a90e1abfe
md"""The new step-length is $1$. There are no blocking constraints
on this step, so the next working set in unchanged: $W_{5}=\left\{ 1\right\} $. The new iterate is $\vec{x}^{\left(5\right)}=\begin{pmatrix}1.4 & 1.7\end{pmatrix}^\mathsf{T}$."""

# ╔═╡ a84051f4-f373-4a25-a74e-77eb7bd6a938
x₄ = x₃ + α₄ .* d₄

# ╔═╡ 89a4d4a0-b976-4b3a-b50b-e4819ecd1f3b
sol₅ = let x = x₄
	g = Q*x - c
	ain = reshape(A[1,:], 1, 2)
	sol = [Q transpose(ain)
		   ain zeros(1,1)] \ [-g; 0]
end

# ╔═╡ d4b39440-d84e-4a2e-a0a0-181cd1c938cf
md"""Finally, we solve the subproblem for $k=5$ to obtain a solution $\vec{d}^{\left(5\right)}=\vec{0}$.
We find a multiplier $\mu_{1}^{\circ}=0.8$, so we have found the
solution. Wet set $\vec{x}^{\star}=\begin{pmatrix}1.4 & 1.7\end{pmatrix}^\mathsf{T}$
and terminate.

The figure below illustrates the iterative process:"""

# ╔═╡ 3bd01fe3-8808-4cc0-83ef-7c3965c40796
let
	x = -1:0.05:5
	y = -1:0.05:5
	z = Surface((x,y)->(0.5 .* [x y]*Q*[x;y] .- transpose(c)*[x;y] .+ 0.5 .* transpose(c)*Q*c)[1], x, y)
	contour(x, y, z, levels=35)
	plot!(x, (2 .+ x) ./ 2, linestyle=:dash, label=L"""-x_1 + 2x_2 - 2 ≤ 0""")
	plot!(x, (6 .- x) ./ 2, linestyle=:dash, label=L""" x_1 + 2x_2 - 6 ≤ 0""")
	plot!(x, (2 .- x) ./ -2, linestyle=:dash, label=L"""x_1 - 2x_2 - 2 ≤ 0""")
	plot!([0,2,4,2,0,0],[0,0,1,2,1,0], linewidth=2, label="feasible domain")
	plot!([2,2,1,1,1,1.4], [0,0,0,0,1.5,1.7], linewidth=2, label="iterations", color=:red, markershape=:circle, xlabel=L"x_1", ylabel=L"x_2", title=L"f : (x_1 - 1)^2 + (x_2 - 2.5)^2")
end

# ╔═╡ d09a2da2-6f35-40cd-b41d-cea42e837629
md"""
##### Sequential Quadratic Programming

We consider the general constrained problem
```math
\begin{aligned}
\min\, & f\left(\vec{x}\right)\\
\textrm{subject to} & \begin{cases}
\vec{h}\left(\vec{x}\right)=\vec{0}\\
\vec{g}\left(\vec{x}\right)\leq\vec{0}
\end{cases}
\end{aligned}
```
where $f:\mathbb{R}^{n}\rightarrow\mathbb{R}$, $\vec{h}:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$,
$m\leq n$, and $\vec{g}:\mathbb{R}^{n}\rightarrow\mathbb{R}^{p}$. The idea
behind the _sequential quadratic programming_ (SQP) approach
is to model the general problem at the current iterate $\vec{x}^{\left(k\right)}$
by a quadratic programming subproblem, then use the minimizer of this
subproblem to define a new iterate $\vec{x}^{\left(k+1\right)}$.
The challenge is to design the quadratic subproblem so that it yields
a good step for the general optimization problem.

We know that the extended Lagrangian function for this problem is

```math
\mathcal{L}\left(\vec{x},\vec{\lambda},\vec{\mu}\right)=f\left(\vec{x}\right)+\vec{\lambda}^\mathsf{T}\vec{h}\left(\vec{x}\right)+\vec{\mu}^\mathsf{T}\vec g\left(\vec{x}\right)\,.
```

Applying Newton's method to the Lagrangian function and linearizing
both the equality and the inequality constraints yields the following
subproblem
```math
\begin{aligned}
\min_{\vec{d}^{\left(k\right)}}\, & \frac{1}{2}\left(\vec{d}^{\left(k\right)}\right)^\mathsf{T} \mathsf{H} \mathcal{L}\left(\vec{x}^{\left(k\right)},\vec{\lambda}^{\left(k\right)},\vec{\mu}^{\left(k\right)}\right)\vec{d}^{\left(k\right)}+ \mathsf{D} \mathcal{L}\left(\vec{x}^{\left(k\right)},\vec{\lambda}^{\left(k\right)},\vec{\mu}^{\left(k\right)}\right)\vec{d}^{\left(k\right)}\\
\textrm{subject to}\, & \begin{cases}
 \mathsf{J}\vec{h}\left(\vec{x}^{\left(k\right)}\right)\vec{d}^{\left(k\right)}=-\vec{h}\left(\vec{x}^{\left(k\right)}\right)\\
 \mathsf{D} g_{j}\left(\vec{x}^{\left(k\right)}\right)\vec{d}^{\left(k\right)}=-g_{j}\left(\vec{x}^{\left(k\right)}\right)\,, & j\in W_{k}\,,
\end{cases}
\end{aligned}
```

where $\mu_{j}^{\left(k\right)}=0$, for all
$j\notin W_{k}$. We can use the active-set method for quadratic programming
to solve this subproblem. The new iterate is given by $\vec{x}^{\left(k+1\right)}$,
$\vec{\lambda}^{\left(k+1\right)}$, $\vec{\mu}^{\left(k+1\right)}$
and $W_{k+1}$.

If the SQP method is able to identify the optimal active set then
it will act like a Newton method for equality-constrained optimization
and will converge rapidly.

It is also remarkable that, far from the solution, the SQP approach
is usually able to improve the estimate of the active set and guide
the iterates towards a solution.

Non-quadratic objective functions, however, can impede progress of
the SQP algorithm, a phenomenon known as the Maratos effect. Steps
that make good progress toward a solution are rejected and the algorithm
fails to converge rapidly. These difficulties can be overcome by means
of a _second-order correction_.
"""

# ╔═╡ 7c41eb45-1ceb-4b4e-8ac9-4c40dd6d386c
md"""
### Interior point methods

In the 1980s it was discovered that many large linear programs could
be solved efficiently by using formulations and algorithms from nonlinear
programming. One characteristic of these methods was that they required
all iterates to satisfy the inequality constraints in the problem
strictly, so they became known as _interior-point methods_. In
the 1990s, a subclass of interior-point methods known as _primal-dual
methods_ had distinguished themselves as the most efficient practical
approaches, and proved to be strong competitors to the _simplex
method_ on large problems. Recently, it has been shown that
interior-point methods are as successful for nonlinear optimization
as for linear programming. General primal-dual interior point methods
are the focus of this section.


For simplicity, we restrict our attention to convex quadratic programs,
which we write as follows:
```math
\begin{aligned}
\min_{\vec{x}}\, & f\left(\vec{x}\right)\overset{\vartriangle}{=}\frac{1}{2} \vec{x}^\mathsf{T}Q\vec{x}- \vec{c}^\mathsf{T}\vec{x}\\
\textrm{subject to}\, & \begin{cases}
A_{\textrm{eq}}\vec{x}=\vec{b}_{\textrm{eq}}\,,\\
A_{\textrm{in}}\vec{x}\leq\vec{b}_{\textrm{in}}\,,
\end{cases}
\end{aligned}
```
where ``Q`` is a symmetric and positive semidefinite ``n\times n`` matrix,
``\vec{c}\in\mathbb{R}^{n}``, ``A_{\textrm{eq}}`` is a ``m\times n`` matrix,
``\vec{b}_{\textrm{eq}}\in\mathbb{R}^{m}``, ``A_{\textrm{in}}`` is a ``p\times n``
matrix and ``\vec{b}_{\textrm{in}}\in\mathbb{R}^{p}``. Rewriting the KKT
conditions in this notation, we obtain
```math
\begin{aligned}
\vec{\mu} & \geq\vec{0}\,,\\
Q\vec{x}+ A_{\textrm{eq}}^\mathsf{T}\vec{\lambda}+ A_{\textrm{in}}^\mathsf{T}\vec{\mu}-\vec{c} & =\vec{0}\,,\\
\left(A_{\textrm{in}}\vec{x}-\vec{b}_{\textrm{in}}\right)_{j}\mu_{j} & =0\,,\quad j=1,\dots,p\,,\\
A_{\textrm{eq}}\vec{x} & =\vec{b}_{\textrm{eq}}\,,\\
A_{\textrm{in}}\vec{x} & \leq\vec{b}_{\textrm{in}}\,.
\end{aligned}
```
By introducing the slack vector ``\vec{y}\geq\vec{0}``, we can rewrite
these conditions as
```math
\begin{aligned}
\left(\vec{\mu},\vec{y}\right) & \geq\vec{0}\,,\\
Q\vec{x}+ A_{\textrm{eq}}^\mathsf{T}\vec{\lambda}+ A_{\textrm{in}}^\mathsf{T}\vec{\mu}-\vec{c} & =\vec{0}\,,\\
y_{j}\mu_{j} & =0\,,\quad j=1,\dots,p\,,\\
A_{\textrm{eq}}\vec{x}-\vec{b}_{\textrm{eq}} & =\vec{0}\,,\\
A_{\textrm{in}}\vec{x}-\vec{b}_{\textrm{in}}+\vec{y} & =\vec{0}\,.
\end{aligned}
```
Since we assume that ``Q`` is positive semidefinite, these KKT conditions
are not only necessary but also sufficient, so we can solve the convex
quadratic program by finding solutions of this system.

Primal-dual methods generate iterates that satisfy the bounds strictly; that is, ``\vec{y}>0``
and ``\vec{\mu}>0``. This property is the origin of the term interior-point.
By respecting these bounds, the methods avoid spurious solutions,
points that satisfy the system but not the bounds. Spurious solutions
abound, and do not provide useful information about real solutions,
so it makes sense to exclude them altogether. Given a current iterate
``\left(\vec{x}^{\left(k\right)},\vec{y}^{\left(k\right)},\vec{\lambda}^{\left(k\right)},\vec{\mu}^{\left(k\right)}\right)``
that satisfies ``\left(\vec{\mu}^{\left(k\right)},\vec{y}^{\left(k\right)}\right)>0``,
we can define a _complementary measure_
```math
\nu_{k}=\frac{ \left(\vec{y}^{\left(k\right)}\right)^\mathsf{T}\vec{\mu}^{\left(k\right)}}{p}\,.
```
This measure gives an indication of the desirability of the couple
``\left(\vec{\mu}^{\left(k\right)},\vec{y}^{\left(k\right)}\right)``.

We derive a path-following, primal-dual method by considering the
perturbed KKT conditions by
```math
\vec{F}\left(\vec{x}^{\left(k+1\right)},\vec{y}^{\left(k+1\right)},\vec{\lambda}^{\left(k+1\right)},\vec{\mu}^{\left(k+1\right)},\sigma_{k}\nu_{k}\right)=\begin{pmatrix}Q\vec{x}^{\left(k+1\right)}+ A_{\textrm{eq}}^\mathsf{T}\vec{\lambda}^{\left(k+1\right)}+ A_{\textrm{in}}^\mathsf{T}\vec{\mu}^{\left(k+1\right)}-\vec{c}\\
Y_{k+1}M_{k+1}\vec{1}-\sigma_{k}\nu_{k}\vec{1}\\
A_{\textrm{eq}}\vec{x}^{\left(k+1\right)}-\vec{b}_{\textrm{eq}}\\
A_{\textrm{in}}\vec{x}^{\left(k+1\right)}-\vec{b}_{\textrm{in}}+\vec{y}^{\left(k+1\right)}
\end{pmatrix}=\vec{0}\,,
```
where
```math
Y_{k+1}=\begin{pmatrix}y_{1}^{\left(k+1\right)} & 0 & \cdots & 0\\
0 & y_{2}^{\left(k+1\right)} & \ddots & 0\\
\vdots & \ddots & \ddots & 0\\
0 & 0 & 0 & y_{p}^{\left(k+1\right)}
\end{pmatrix}\,,\quad M_{k+1}=\begin{pmatrix}\mu_{1}^{\left(k+1\right)} & 0 & \cdots & 0\\
0 & \mu_{2}^{\left(k+1\right)} & \ddots & 0\\
\vdots & \ddots & \ddots & 0\\
0 & 0 & 0 & \mu_{p}^{\left(k+1\right)}
\end{pmatrix}\,,
```
and ``\sigma\in\left[0,1\right]`` is the reduction factor that we wish
to achieve in the complementary measure on one step. We call ``\sigma``
the _centering parameter_. The solution of this system for all
positive values of ``\sigma`` and ``\nu`` define the _central path_,
which is a trajectory that leads to the solution of the quadratic
program as ``\sigma\nu`` tends to zero.

By fixing ``\sigma_{k}`` and applying Newton's method to the system,
we obtain the linear system
```math
\begin{pmatrix}Q & 0 &  A_{\textrm{eq}}^\mathsf{T} &  A_{\textrm{in}}^\mathsf{T}\\
0 & M_{k} & 0 & Y_{k}\\
A_{\textrm{eq}} & 0 & 0 & 0\\
A_{\textrm{in}} & I & 0 & 0
\end{pmatrix}\begin{pmatrix}\vec{d}_{\vec{x}}^{\left(k\right)}\\
\vec{d}_{\vec{y}}^{\left(k\right)}\\
\vec{d}_{\vec{\lambda}}^{\left(k\right)}\\
\vec{d}_{\vec{\mu}}^{\left(k\right)}
\end{pmatrix}=-\begin{pmatrix}Q\vec{x}^{\left(k\right)}+ A_{\textrm{eq}}^\mathsf{T}\vec{\lambda}^{\left(k\right)}+ A_{\textrm{in}}^\mathsf{T}\vec{\mu}^{\left(k\right)}-\vec{c}\\
Y_{k}M_{k}\vec{1}-\sigma_{k}\nu_{k}\vec{1}\\
A_{\textrm{eq}}\vec{x}^{\left(k\right)}-\vec{b}_{\textrm{eq}}\\
A_{\textrm{in}}\vec{x}^{\left(k\right)}-\vec{b}_{\textrm{in}}+\vec{y}^{\left(k\right)}
\end{pmatrix}\,.
```
We obtain the next iterate by setting
```math
\begin{pmatrix}\vec{x}^{\left(k+1\right)}\\
\vec{y}^{\left(k+1\right)}\\
\vec{\lambda}^{\left(k+1\right)}\\
\vec{\mu}^{\left(k+1\right)}
\end{pmatrix}=\begin{pmatrix}\vec{x}^{\left(k\right)}\\
\vec{y}^{\left(k\right)}\\
\vec{\lambda}^{\left(k\right)}\\
\vec{\mu}^{\left(k\right)}
\end{pmatrix}+\alpha_{k}\begin{pmatrix}\vec{d}_{\vec{x}}^{\left(k\right)}\\
\vec{d}_{\vec{y}}^{\left(k\right)}\\
\vec{d}_{\vec{\lambda}}^{\left(k\right)}\\
\vec{d}_{\vec{\mu}}^{\left(k\right)}
\end{pmatrix}\,,
```
where ``\alpha_{k}`` is chosen to retain the bounds ``\left(\vec{\mu}^{\left(k+1\right)},\vec{y}^{\left(k+1\right)}\right)>0``
and possibly to satisfy various other conditions.

The choices of centering parameter ``\sigma_{k}`` and step-length ``\alpha_{k}``
are crucial for the performance of the method. Techniques for controlling
these parameters, directly and indirectly, give rise to a wide variety
of methods with diverse properties. One option is to use equal step
length for the primal and dual updates, and to set ``\alpha_{k}=\min\left\{ \alpha_{k}^{\textrm{pri}},\alpha_{k}^{\textrm{dual}}\right\} ``,
where
```math
\begin{aligned}
\alpha_{k}^{\textrm{pri}} & =\max\left\{ \alpha\in\left\{ 0,1\right\} :\vec{y}^{\left(k\right)}+\alpha\vec{d}_{\vec{y}}^{\left(k\right)}\geq\left(1-\tau\right)\vec{y}^{\left(k\right)}\right\} \,,\\
\alpha_{k}^{\textrm{dual}} & =\max\left\{ \alpha\in\left\{ 0,1\right\} :\vec{\mu}^{\left(k\right)}+\alpha\vec{d}_{\vec{\mu}}^{\left(k\right)}\geq\left(1-\tau\right)\vec{\mu}^{\left(k\right)}\right\} \,,
\end{aligned}
```
the parameter ``\tau\in\left]0,1\right[`` controls how far we back
off from the maximum step for which the conditions ``\vec{y}^{\left(k\right)}+\alpha\vec{d}_{\vec{y}}^{\left(k\right)}\geq\vec{0}``
and ``\vec{\mu}^{\left(k\right)}+\alpha\vec{d}_{\vec{\mu}}^{\left(k\right)}\geq\vec{0}``
are satisfied. A typical value of ``\tau=0.995`` and we can choose
``\tau_{k}`` to approach ``1`` as the iterates approach the solution,
to accelerate the convergence.

The most popular interior-point method for convex QP is based on Mehrotra's
predictor-corrector. First we compute an affine scaling step ``\left(\vec{d}_{\vec{x},\textrm{aff}},\vec{d}_{\vec{y},\textrm{aff}},\vec{d}_{\vec{\lambda},\textrm{aff}},\vec{d}_{\vec{\mu},\textrm{aff}}\right)``
by setting ``\sigma_{k}=0``. We improve upon this step by computing
a corrector step. Next, we compute the centering parameter ``\sigma_{k}``
using following heuristic
```math
\sigma_{k}=\left(\frac{\nu_{\textrm{aff}}}{\nu_{k}}\right)^{3}\,,
```
where ``\nu_{\textrm{aff}}=\frac{ \left(\vec{y}_{\textrm{aff}}\right)^\mathsf{T}\left(\vec{\mu}_{\textrm{aff}}\right)}{p}``.
The total step is obtained by solving the following system
```math
\begin{pmatrix}Q & 0 &  A_{\textrm{eq}}^\mathsf{T} &  A_{\textrm{in}}^\mathsf{T}\\
0 & M_{k} & 0 & Y_{k}\\
A_{\textrm{eq}} & 0 & 0 & 0\\
A_{\textrm{in}} & I & 0 & 0
\end{pmatrix}\begin{pmatrix}\vec{d}_{\vec{x}}^{\left(k\right)}\\
\vec{d}_{\vec{y}}^{\left(k\right)}\\
\vec{d}_{\vec{\lambda}}^{\left(k\right)}\\
\vec{d}_{\vec{\mu}}^{\left(k\right)}
\end{pmatrix}=-\begin{pmatrix}Q\vec{x}^{\left(k\right)}+ A_{\textrm{eq}}^\mathsf{T}\vec{\lambda}^{\left(k\right)}+ A_{\textrm{in}}^\mathsf{T}\vec{\mu}^{\left(k\right)}-\vec{c}\\
Y_{k}M_{k}\vec{1}+\Delta Y_{\textrm{aff}}\Delta M_{\textrm{aff}}\vec{1}-\sigma_{k}\nu_{k}\vec{1}\\
A_{\textrm{eq}}\vec{x}^{\left(k\right)}-\vec{b}_{\textrm{eq}}\\
A_{\textrm{in}}\vec{x}^{\left(k\right)}-\vec{b}_{\textrm{in}}+\vec{y}^{\left(k\right)}
\end{pmatrix}\,,
```
where
```math
\Delta Y_{\textrm{aff}}=Y_{\textrm{aff}}-Y_{k}\,,\quad\Delta M_{\textrm{aff}}=M_{\textrm{aff}}-M_{k}\,.
```

"""

# ╔═╡ b6fe2edd-721e-4ff4-9aae-8b281e366b11
md"""
#### Implementations
We will use the [Ipopt solver](https://coin-or.github.io/Ipopt/), which is written in C. It is made available to the Julia language through the [`Ipopt.jl`](https://github.com/jump-dev/Ipopt.jl) package, which integrates perfectly with the JuMP framework, making it very user-friendly.

Considering the example from earlier, i.e.
```math
\begin{aligned}
\min_{\vec{x}}\, & f\left(\vec{x}\right)=\left(x_{1}-1\right)^{2}+\left(x_{2}-2.5\right)^{2}\\
\textrm{subject to}\, & \begin{cases}
-x_{1}+2x_{2}-2\leq0\,,\\
x_{1}+2x_{2}-6\leq0\,.\\
x_{1}-2x_{2}-2\leq0\,,\\
-x_{1}\leq0\,,\\
-x_{2}\leq0\,.
\end{cases}
\end{aligned}
```
You can see in the code below, that the implementation is an almost direct copy of the original problem. 
"""

# ╔═╡ c50998b1-94b6-4606-b138-a95dcbf4f56c
let
	model = Model(Ipopt.Optimizer)
	set_attribute(model, "print_level", 0) # to limit the solver's output
	@variable(model, x₁ ≥ 0, start=2)
	@variable(model, x₂ ≥ 0, start=0)
	@NLobjective(model, Min, (x₁-1)^2+(x₂-2.5)^2)
	@constraint(model, con1, -x₁+2x₂ <= 2)
	@constraint(model, con2,  x₁+2x₂ <= 6)
	@constraint(model, con3,  x₁-2x₂ <= 2)
	optimize!(model)
    @info """Obtained solution: 
	- x₁: $(value(x₁))
	- x₂: $(value(x₂))
	"""
end

# ╔═╡ Cell order:
# ╟─05653593-4365-46a2-8e1b-79c3c8e11575
# ╟─e0115026-3eb8-4127-b275-34c9013fed2c
# ╠═b8671dde-ef61-4e07-8243-4d9521d2fcd0
# ╟─60e010dc-3669-424e-b001-085faa6b93ba
# ╟─b02cadd0-65d0-4395-ba64-35f023696ce7
# ╟─7a9fa9f1-b50b-4e08-b5c3-537226e3deda
# ╟─88447980-da25-404f-abcf-29927d3c0168
# ╟─94ac3d2e-6f8f-45ef-b9ba-93b3af11098a
# ╟─163c7e36-7394-4550-92a6-5a263cf69e09
# ╟─b67154e3-256e-44e2-be84-ac3f2462d58a
# ╟─d75a7f24-e407-4585-b9ce-0e2d13e0dba1
# ╟─5ac0b0d5-ce4d-4851-9691-c557e5b8fb90
# ╠═a6ff71ff-522f-4f41-bb2c-edd8fc944024
# ╟─a95111dd-27c8-4d66-a278-ca58badf1111
# ╠═167d22c7-a625-4dc3-98cd-fcf8a1789c25
# ╟─6ff21901-4055-4221-88ed-f1a3aec2f2ce
# ╠═bb7ebc81-676d-41a9-b119-95b84655378b
# ╟─7da441d0-3a9c-4ce0-80f1-508f8da944f6
# ╠═fa47ae35-f707-494c-8fb6-ef65076f13c4
# ╟─5349edce-28c7-4f65-a293-2a7059df2e60
# ╠═7f2110fa-a514-421e-9d13-57400f56c63b
# ╟─e1a856c5-2b03-47aa-87f0-7fcb952cc2e3
# ╟─8cb2b3ae-1ebf-4694-adf1-42c1c7863016
# ╟─b4ed8968-fc9d-4f98-8efd-656a6a0a7acf
# ╠═3c5b411c-872f-4d67-b307-d686d649b129
# ╟─98be776a-108c-4bc5-9095-30e8756ff048
# ╟─a99cffd7-d37c-47b2-82c2-f57cf90f3d95
# ╟─060516fe-0b94-4278-9f81-913d336c67d1
# ╠═b0601aae-10ff-4405-b498-6649a0036e1e
# ╟─d07169d1-3605-449e-a3fd-c2646e876da9
# ╠═f40dced6-78c6-4ad0-88b1-85896573f330
# ╟─ed99bd93-bb76-4f18-af4b-c3439b40d3e0
# ╠═7dcb0c9a-8036-4424-ab97-d7a909dbcb02
# ╟─6a727c57-e605-4813-aef8-b23766daacda
# ╠═5960fea9-007d-4151-9354-a1f215a8987e
# ╟─6e6423b6-52fa-43e0-b556-03c670ed9b7a
# ╠═c6854393-3671-4858-b1ec-92f77a291b6a
# ╟─21f11662-4da4-47d1-af0b-f82b167f9708
# ╠═fe731655-3319-4009-b5d5-778af69d9198
# ╟─b2699452-d0fb-443a-bf6c-7d8c10a1717a
# ╠═1c05c58c-f73d-499f-81d9-c071bb931362
# ╟─282c0f80-015b-11eb-19d8-d9235a2a77d4
# ╟─4d9273a0-015a-11eb-2caa-130d0245a946
# ╟─bfe6caf0-015a-11eb-23a0-919e0520bd8d
# ╟─0aa85db0-015b-11eb-01ae-694c21b1f7a4
# ╟─c6569c40-016d-11eb-2d6b-4d8a54cff763
# ╟─0e6ae94e-016e-11eb-3f19-676f24836ebb
# ╟─6bf23d80-016e-11eb-1c64-6d2da5f0f954
# ╟─7cf25a20-016e-11eb-30f1-8394316a43c2
# ╟─c5157440-016e-11eb-33fd-4b9c155cf832
# ╟─fce70140-016e-11eb-224a-3d863b6817b6
# ╟─18554920-0186-11eb-0cb2-f5d020aff0d9
# ╟─392084d2-0186-11eb-11e5-2b5128129e19
# ╟─6e95360e-0186-11eb-1ed3-770285bc0e74
# ╟─86bb011e-0186-11eb-2b14-8ddd7f9ca6c2
# ╠═202572f0-0187-11eb-0172-9bea075d89a9
# ╟─d332f672-0186-11eb-205a-97aeef39b75d
# ╟─4680a1e0-0187-11eb-3a9c-5b631ea7c822
# ╠═5815d290-0187-11eb-391d-ff073eb9dc85
# ╟─0589ddc0-0c2d-42dc-b526-6334144b4476
# ╠═75d97390-0187-11eb-1b3f-69c879780178
# ╟─254bd98d-db2c-4eb8-9bb9-85e574420254
# ╟─7927b430-0187-11eb-345f-a10499db73fa
# ╠═86f5f1d0-0187-11eb-1186-ad0df4a1ef6c
# ╟─8dd8f1f0-0187-11eb-370a-a1b5053e91e4
# ╠═97810580-0187-11eb-38a3-fd83c085e89c
# ╟─9fb64f30-0187-11eb-1a26-7f9e577886d9
# ╠═ad0c2880-0187-11eb-31c1-bdfd311b2650
# ╟─ba3db0f0-0187-11eb-2efa-296cff31defa
# ╠═c818e6e0-0187-11eb-2c4a-b78c74a10231
# ╟─0e4f381b-e6bc-4be3-a18b-d22aa1a5fc73
# ╟─cd8db100-0187-11eb-0ac7-2f4510dc2f08
# ╠═d80597b0-0187-11eb-23ef-7daa3ac30f36
# ╟─e07a5b60-0187-11eb-246d-f18cbf91b1f1
# ╠═f2674900-0187-11eb-017d-e5d26651604d
# ╟─f933b3e0-0187-11eb-113f-876918888691
# ╟─2837d5e2-0188-11eb-1eb3-67c8aeb366d0
# ╟─35b66ce0-0188-11eb-2b93-fb6f4e8fe9ec
# ╟─648f6030-0188-11eb-2ca2-7b2e080de91b
# ╟─3c2d2158-d160-49ec-9030-bf39b7ec75d4
# ╟─ad19fcc2-c149-40d6-b2eb-c112312f066a
# ╠═a0b5b8b0-14e0-4a50-abec-74f9130feed2
# ╟─1af9b16a-b9dd-4bcb-bc9d-035b7cf18568
# ╟─9256f823-1ea5-4dc7-86e4-7cd85c07433d
# ╠═9a1cf74e-ce12-45ce-9e54-b8eddb7d4d83
# ╟─714ca8fe-ab51-4ecf-a5a8-8193fc3d8a28
# ╟─fba69647-7a8a-4c3f-aa81-b743d72e7a99
# ╟─a26bac7f-d605-4285-b4bc-40b5e7da8745
# ╠═0307b650-1d50-4348-9272-d5575d5308d4
# ╠═9259ba81-2026-4781-b438-e356b5cd4e4e
# ╟─c636a610-9157-42c0-8bda-767112b4b475
# ╠═781240f7-0323-4be3-960e-6960902f3aaa
# ╟─7f09eb59-4c95-40a5-9858-1668c3c187b4
# ╠═a8ae98e2-59d1-45d3-a824-6072203465d9
# ╟─bd8a4b3f-86e5-4b32-847b-a2cf0ed6af04
# ╠═61780b00-1d06-4e8a-af35-3e0340d9fb5b
# ╠═fded31f0-d136-4c6f-949b-40b2027b45ad
# ╟─aeec3f14-f31f-455c-bd5c-1491b692b161
# ╠═85d2c8cf-418d-4709-b3bc-df2dce39a1d2
# ╠═9308ce1a-0fa5-4550-a203-48d8ea66f5ad
# ╟─591f1052-d1e5-4078-b010-eb78d69b7789
# ╠═6224ef69-0ecc-4bb1-8295-53335a719681
# ╟─e5cd3218-34d4-4b8d-b8ad-428110409acd
# ╠═d57a4395-d2ea-4480-bbad-127e082378e6
# ╠═4f35e84d-871b-4418-b079-b61dcd05bcce
# ╟─77d30dfa-3538-4111-ae3d-b506bea0d81f
# ╠═e3eec868-7191-4296-8aaa-bd61670b00da
# ╟─612c70a9-c220-44ad-aff6-569a90e1abfe
# ╠═a84051f4-f373-4a25-a74e-77eb7bd6a938
# ╠═89a4d4a0-b976-4b3a-b50b-e4819ecd1f3b
# ╟─d4b39440-d84e-4a2e-a0a0-181cd1c938cf
# ╟─3bd01fe3-8808-4cc0-83ef-7c3965c40796
# ╟─d09a2da2-6f35-40cd-b41d-cea42e837629
# ╟─7c41eb45-1ceb-4b4e-8ac9-4c40dd6d386c
# ╟─b6fe2edd-721e-4ff4-9aae-8b281e366b11
# ╠═c50998b1-94b6-4606-b138-a95dcbf4f56c
