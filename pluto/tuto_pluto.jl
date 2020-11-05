### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 31952da6-1f0e-11eb-2266-85a97813c35c
begin
using Gridap

#First, we describe the domain Ω
domain = (0,1,0,1)
n = 2^7
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

#Then, we describe the spaces where the variables are living. There are two types of functions. Test functions:
V0 = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")
#and the "solution" function:
g(x) = 0.0
Ug = TrialFESpace(V0,g) #Here the function g corresponds to the Dirichlet condition.

trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

#Now, we describe the equation we want to solve.
#Each integral is described with a FETerm.
#In the linear case, we can use AffineFETerm.
w(x) = 1.0
a(u,v) = ∇(v)⊙∇(u)
b_Ω(v) = v*w
t_Ω = AffineFETerm(a,b_Ω,trian,quad)

#Terms are aggregated in a FEOperator:
op_edp = AffineFEOperator(Ug,V0,t_Ω)
	
ls = LUSolver()
solver = LinearFESolver(ls)
uh = solve(solver,op_edp)
end

# ╔═╡ 156e53ec-1f12-11eb-2766-8f32aadc1b9e
begin
	
	include("src/PDENLPModels.jl")

	using Main.workspace190.PDENLPModels

end

# ╔═╡ a5ff4dbe-1f06-11eb-39a6-cbd2ca56ade2
md"## How to Use GridapPDENLPModels
In this short tutorial, we show how to use GridapPDENLPModels an NLPModels for optimization problems with PDE-constraints that uses [Gridap.jl](https://github.com/gridap/Gridap.jl) to discretize the PDE with finite elements."

# ╔═╡ f3b870ce-1f08-11eb-298e-df588db76ef4
md"
In a simplified setting, we look for two functions
```math
y:\Omega \rightarrow \mathbb{R}^{n_y}, u:\Omega \rightarrow \mathbb{R}^{n_u}
``` 
satisfying
"

# ╔═╡ b8499954-1f07-11eb-20d1-9f96b605a016
md"
```math
\begin{equation}
	\begin{aligned}
	\min_{y \in Y_{edp},u \in Y_{con} } \  & \int_{\Omega} f(y,u)d\Omega, \\
	\mbox{ s.t. } & y \mbox{ sol. of } \mathrm{EDP}(u) \mbox{ (+ boundary conditions on $\partial\Omega$)},
	\end{aligned}
	\end{equation}
```"

# ╔═╡ 91c7e358-1f09-11eb-2d64-eb8adb028644
md"
For instance, consider the control of a membrane under a force f:
```math
\begin{equation}
	\begin{array}{clc}
	\min\limits_{y(x) \in \mathbb{R},u(x) \in \mathbb{R} } \  & \int_{\Omega} |y - y_d(x)|^2 + \alpha|v|^2dx, \\
	\mbox{ s.t. } & -\Delta y(x) = f(x) + u(x), &\quad x \in \Omega,\\
                  & y(x) = 0, &\quad x \in \partial \Omega.
	\end{array}
	\end{equation}
```
``
\text{where } y \text{ is the vertical movement of the membrane}, v \text{ is a control force, } y_d
``
``\text{ is a target function, and } \alpha >0 \text{ some constant}.
``
In our example, we will choose ``$\Omega=(0,1)^2$``
In order to apply FE methods, we use the weak formulation of the PDE, in this case:
```math
\begin{equation}
	\begin{array}{clc}
	\min\limits_{y(x) \in \mathbb{R},u(x) \in \mathbb{R} } \  & \int_{\Omega} |y - y_d(x)|^2 + \alpha|v|^2dx, \\
	\mbox{ s.t. } & \int_{\Omega}\nabla y ⋅ \nabla vdx = \int_{\Omega}(f + u)vdx, &\quad \text{ in } \Omega, \forall v \in V,\\
                  & y = 0, &\quad \text{ on } \partial \Omega.
	\end{array}
	\end{equation}
```
"

# ╔═╡ 965fc458-1f13-11eb-0ecf-53154f37c23a
md"First, for a fixed control, let us see how we can use Gridap to solve the PDE."

# ╔═╡ 929926b0-1f0f-11eb-3fa0-8d5815006d72
md" We can proceed in a similar way, to define our GridapPDENLPModel, by:
* Setting the Test and Trial spaces;
* Model the constraints with an FEOperator;
* Model the objective function in a similar way.
" 

# ╔═╡ d55fcdfa-1f0f-11eb-3bec-c7af8c6a7cee
begin
	
	Yedp = Ug
	Xedp = V0
		Xcon = TestFESpace(
			reffe=:Lagrangian, order=1, valuetype=Float64,
			conformity=:H1, model=model)
	Ycon = TrialFESpace(Xcon)

	nothing;
end

# ╔═╡ e520d750-1f11-11eb-2be9-bdd6b1e73125
begin

	Y = MultiFieldFESpace([Yedp, Ycon])

	ybis(x) =  -1#(x[1]-1)^2+(x[2]-1)^2
	function f(yu)
		y, u = yu
		0.5 * (ybis - y) * (ybis - y) + 0.5 * u * u
	end

	function res_Ω(yu, v)
	 y, u = yu
	 v

	 -∇(v)⊙∇(y) - v*u
	end
	wf(x) = 1.0
	b(v) = v*wf
	term = AffineFETerm(res_Ω, b, trian, quad)
	op = FEOperator(Y, V0, t_Ω)

end

# ╔═╡ 08451c3e-1f12-11eb-23c2-e93f32e5980e
begin

	nvar = Gridap.FESpaces.num_free_dofs(Y)
	xin  = zeros(nvar)
	
	nlp = GridapPDENLPModel(xin, zeros(0), f, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)
	#Tanj: try this one:
	#nlp = GridapPDENLPModel(xin, f, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)
	@show "nvar= ", nlp.meta.nvar

end

# ╔═╡ 6894eb7c-1f07-11eb-0125-39ed0c8d4b53
md"## References

* Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia. Journal of Open Source Software, 5(52), 2520.

"

# ╔═╡ Cell order:
# ╟─a5ff4dbe-1f06-11eb-39a6-cbd2ca56ade2
# ╟─f3b870ce-1f08-11eb-298e-df588db76ef4
# ╟─b8499954-1f07-11eb-20d1-9f96b605a016
# ╠═91c7e358-1f09-11eb-2d64-eb8adb028644
# ╟─965fc458-1f13-11eb-0ecf-53154f37c23a
# ╠═31952da6-1f0e-11eb-2266-85a97813c35c
# ╠═929926b0-1f0f-11eb-3fa0-8d5815006d72
# ╠═d55fcdfa-1f0f-11eb-3bec-c7af8c6a7cee
# ╠═e520d750-1f11-11eb-2be9-bdd6b1e73125
# ╠═156e53ec-1f12-11eb-2766-8f32aadc1b9e
# ╠═08451c3e-1f12-11eb-23c2-e93f32e5980e
# ╠═6894eb7c-1f07-11eb-0125-39ed0c8d4b53
