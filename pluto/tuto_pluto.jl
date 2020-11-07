### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 31952da6-1f0e-11eb-2266-85a97813c35c
begin
	using Gridap

	#First, we describe the domain Ω
	domain = (0,1,0,1)
	n = 2^5
	partition = (n,n)
	model = CartesianDiscreteModel(domain,partition)

	#and define the machinery to compute integrals
	trian = Triangulation(model) #integration mesh
	degree = 2
	quad = CellQuadrature(trian,degree)	#quadrature
end

# ╔═╡ 156e53ec-1f12-11eb-2766-8f32aadc1b9e
begin

	using Main.workspace3.PDENLPModels
end

# ╔═╡ 1b2edd22-1f82-11eb-26aa-b9db6cb28123
begin
	
	include("../src/PDENLPModels.jl")

end

# ╔═╡ a5ff4dbe-1f06-11eb-39a6-cbd2ca56ade2
md"## How to Use GridapPDENLPModels
In this short tutorial, we show how to use GridapPDENLPModels an NLPModels for optimization problems with PDE-constraints that uses [Gridap.jl](https://github.com/gridap/Gridap.jl) to discretize the PDE with finite elements."

# ╔═╡ f3b870ce-1f08-11eb-298e-df588db76ef4
md"
In a simplified setting, we look for two functions
```math
y:\Omega \rightarrow \mathbb{R}^{n_y}, \text{ and } u:\Omega \rightarrow \mathbb{R}^{n_u}
``` 
satisfying
"

# ╔═╡ b8499954-1f07-11eb-20d1-9f96b605a016
md"
```math
\begin{equation}
	\begin{aligned}
	\min_{y \in Y_{edp},u \in Y_{con} } \  & \int_{\Omega} f(y,u)d\Omega, \\
	\mbox{ s.t. } & y \mbox{ sol. of } \mathrm{PDE}(u) \mbox{ (+ boundary conditions on $\partial\Omega$)}.
	\end{aligned}
	\end{equation}
```"

# ╔═╡ 91c7e358-1f09-11eb-2d64-eb8adb028644
md"
For instance, consider the control of a membrane under a force f:
```math
\begin{equation}
	\begin{array}{clc}
	\min\limits_{y(x) \in \mathbb{R},u(x) \in \mathbb{R} } \  & \int_{\Omega} |y - y_d(x)|^2 + \alpha|u|^2dx, \\
	\mbox{ s.t. } & -\Delta y(x) = w(x) + u(x), &\quad x \in \Omega,\\
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
	\min\limits_{y(x) \in \mathbb{R},u(x) \in \mathbb{R} } \  & \int_{\Omega} |y - y_d(x)|^2 + \alpha|u|^2dx, \\
	\mbox{ s.t. } & \int_{\Omega}\nabla y ⋅ \nabla vdx = \int_{\Omega}(w + u)vdx, &\quad \text{ in } \Omega, \forall v \in V,\\
                  & y = 0, &\quad \text{ on } \partial \Omega.
	\end{array}
	\end{equation}
```
"

# ╔═╡ 965fc458-1f13-11eb-0ecf-53154f37c23a
md"
### Solve a PDE with Gridap
For a fixed control, let us see how we can use Gridap to solve the PDE. **Step 1** we describe the discretized domain Ω and the machinery to compute integrals."

# ╔═╡ 1e0b77d8-1f84-11eb-2a68-77d7c0297260
md"
**Step 2** Then, we describe the spaces where the variables are living. There are two types of functions: *test functions*, and the solution (in Gridap called *trial functions*). At this stage, we also specify the Dirichlet boundary conditions.
"

# ╔═╡ dddbaad2-1f82-11eb-1b16-19dc9fcfe111
begin

	#Then, we describe the spaces where the variables are living. There are two types of functions. Test functions:
	V0 = TestFESpace(
	  reffe=:Lagrangian, 
	  order=1, #first order, Lagrangian interpolation of the function at FE element
	  valuetype=Float64, #the test and trial functions are real-valued
	  conformity=:H1, #this space is a subset of H^1(Ω)
	  model=model, 
	  dirichlet_tags="boundary") #tags refer to a part/region of the domain
	
	#and the "solution" function:
	y0(x) = 0.0
	Ug = TrialFESpace(V0,y0) #The function g corresponds to the Dirichlet condition.
	
end

# ╔═╡ e7ed13fe-1f84-11eb-0101-3dbe71d41f09
md"
**Step 3** Once the framework is set, we define the problem. Each integral is described with a **FETerm**. In our example, we have
``
$a(y,v)= \int_{\Omega}\nabla y ⋅ \nabla v~dx ,\text{ and } b(v) = \int_{\Omega}w v~dx.$
``
"

# ╔═╡ d840767c-1f83-11eb-006b-3f726d9a5168
begin

	#Now, we describe the equation we want to solve.
	#Each integral is described with a FETerm.
	w(x)   = 10.0
	a(y,v) = ∇(v)⊙∇(y)
	b_Ω(v) = v*w
	t_Ω    = AffineFETerm(a,b_Ω,trian,quad) #In the linear case, we use AffineFETerm.

	#Terms are aggregated in a FEOperator:
	op_edp = AffineFEOperator(Ug,V0,t_Ω)
	
end

# ╔═╡ 82d9a9ea-1f85-11eb-010a-a7a170ee5ab9
md"
**Step 4** The problem is now reduced to a linear equation that we can solve with our favorite approach.
"

# ╔═╡ f2e34852-1f82-11eb-0fec-4b84dcc82272
begin

	#Gridap's suggestion to solve the linear problem:	
	ls = LUSolver()
	solver = LinearFESolver(ls)
	uh = solve(solver,op_edp) #the result if a FEFunction
	
	sol_uh = get_free_values(uh) #translate a FEFunction in a vector.
	
	#We can print the residual:
	@show norm(Gridap.FESpaces.residual(op_edp, uh), Inf)
	
end

# ╔═╡ 4fd4f7ec-1f86-11eb-1553-cdf04b28889c
md"
**Step 5** Visualize the solution using Paraview (not in Julia for now).
"

# ╔═╡ 5d72f79e-1f86-11eb-12ba-c3546a5bbed9
begin

	#and print the solution (using Paraview):
	writevtk(trian,"data/results",cellfields=["uh"=>uh])
	
end

# ╔═╡ 929926b0-1f0f-11eb-3fa0-8d5815006d72
md" 
### Solve a (linear) PDE-constrained opimization problem with Gridap and NLPModels
We can proceed in a similar way, to define our GridapPDENLPModel, by:
* Setting the Test and Trial spaces;
* Model the constraints with an FEOperator;
* Model the objective function in a similar way.

**Step 1** We describe the discretized domain Ω and the machinery to compute integrals. We reuse the one defined before.

**Step 2** We define the test and trial spaces. Note that for the control, there is a no Dirichlet boundary conditions.
" 

# ╔═╡ d55fcdfa-1f0f-11eb-3bec-c7af8c6a7cee
begin
	
	Yedp = Ug
	Xedp = V0
	Xcon = TestFESpace(
			reffe=:Lagrangian, 
		    order=1, 
		    valuetype=Float64,
			conformity=:H1, #another possibility would be conformity=:L2
		    model=model) 
	Ycon = TrialFESpace(Xcon)
	
	#It will be convenient to have the whole solution+control space.
	Y = MultiFieldFESpace([Yedp, Ycon])

end

# ╔═╡ 1fbad738-1f87-11eb-1dd3-cb7cbd083b74
md"
**Step 3** Once the framework is set, we define the problem. Each integral is described with a **FETerm**. In our example, we have
``
$a(y,u,v)= \int_{\Omega}\nabla y ⋅ \nabla v - v u~dx ,\text{ and } b(v) = \int_{\Omega}w v~dx.$
``
and
``
$f(y,u) = \int_{\Omega} 1000 * |y - y_d(x)|^2 + |u|^2dx$
``
We have only one type of integral here, so one FETerm.
"

# ╔═╡ e520d750-1f11-11eb-2be9-bdd6b1e73125
begin

	function res_Ω(yu, v)
	 y, u = yu
	 v

	 ∇(v)⊙∇(y) - v*u
	end
	b(v) = v*w
	#Create the FETerm:
	term = AffineFETerm(res_Ω, b, trian, quad)
	#Create the FEOperator:
	op = AffineFEOperator(Y, V0, term)
	
	yd(x) =  -1#(x[1]-1)^2+(x[2]-1)^2
	function f(yu)
		y, u = yu #This is a specific feature of Gridap for MultiFieldFunctions.
		
		1000 * 0.5 * (yd - y) * (yd - y) + 0.5 * u * u
	end

end

# ╔═╡ 7bd0918c-1f89-11eb-022d-3753157b3a34
md"
**Step 4** We are now ready to define a **GridapNLPModel**.
"

# ╔═╡ 08451c3e-1f12-11eb-23c2-e93f32e5980e
begin

	nvar = Gridap.FESpaces.num_free_dofs(Y) #gives the size of the working space in Y
	xin  = zeros(nvar)
	
	nlp = GridapPDENLPModel(xin, #initial guess
		                    f,  #objective function
		                    Yedp, #Trial space of the solution of the PDE
		                    Ycon, #Trial space of the control
		                    Xedp, #Test space of the solution of the PDE
		                    Xcon, #Test space of the control
		                    trian, #integration mesh for the objective function
		                    quad,  #integration quadrature for the objective function
		                    op = op) #FEOperator of the constraints
	
	@show "nvar= ", nlp.meta.nvar

end

# ╔═╡ c2fdf630-1f89-11eb-3ede-1d2e96a49efb
begin
	
	using FastClosures, Krylov, LinearAlgebra, LinearOperators, Test
	
	#This is a quadratic-linear optimizaion problem
	xr = rand(nlp.meta.nvar)
	
	G = Symmetric(hess(nlp, rand(nlp.meta.nvar)),:L)
	g = grad(nlp, zeros(nlp.meta.nvar))
	c = obj(nlp, zeros(nlp.meta.nvar))
	
	f_test = 0.5*xr'*G*xr + g'*xr + c
	fx = obj(nlp, xr)
	
	@show norm(f_test - fx)
	
end

# ╔═╡ a6cc78ba-1f89-11eb-1c7c-4b6d8e45529b
md"
**Remark**: this problem is a quadratic optimization problem with linear constraints.
``
$f(x) = x^TGx + g^Tx + c$
``
where
"

# ╔═╡ b24cf81c-1f8a-11eb-339c-4139a5064049
md"
To solve this linear-quadratic optimization problem, we will use an iterative method for the following symmetric linear system
``
$\left( \begin{array}{cc} G & A^T \\ A & 0 \end{array} \right)\left( \begin{array}{c} x \\ \lambda \end{array} \right) = \left( \begin{array}{c} -g \\ b \end{array} \right)$
``
(not so smart but simple...)
"

# ╔═╡ 3360adfc-1f8a-11eb-3a6e-77dfa01e929d
begin
	function quick_algo_quad_lin_pbs(nlp :: GridapPDENLPModel; 
									 x0  :: AbstractVector{T} = nlp.meta.x0) where T

	 #We solve quasi-definite linear system
	 # ( G  A' ) (x) = (-g)
	 # ( A  0  ) (λ) = ( b)
	 #G = hess(nlp, x0)
	 #A = jac(nlp,  x0)
	 #g = grad(nlp, 0.)
	 #b = cons(nlp, 0.)
	 gx = grad(nlp, zeros(nlp.meta.nvar))
	 c0 = cons(nlp, zeros(nlp.meta.nvar))

	 rhs = vcat(- gx, c0)

	 #We now create a LinearOperator
	 Jv  = Array{T,1}(undef, nlp.meta.ncon)
	 Jtv = Array{T,1}(undef, nlp.meta.nvar)
		
	 prod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jtprod(nlp, x0, v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon]),
							   jprod(nlp, x0, v[1:nlp.meta.nvar]))
	 ctprod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jtprod(nlp, x0, v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon]),
							   jprod(nlp, x0, v[1:nlp.meta.nvar]))
	 #PreallocatedLinearOperator{T}(nrow::Int, ncol::Int, symmetric::Bool, hermitian::Bool, prod, tprod, ctprod)
	 jac_op = PreallocatedLinearOperator{T}(nlp.meta.ncon+nlp.meta.nvar, nlp.meta.ncon+nlp.meta.nvar, true, true, prod, ctprod, ctprod)

	 (x, stats) = minres(jac_op, rhs)

	 return x, stats
	end
end

# ╔═╡ 55fbe488-1f99-11eb-0c13-97a3126e320b
begin
	nlp.meta.ncon
end

# ╔═╡ a11246d0-1f8b-11eb-1273-a795fc9802ab
md"
**Step 5** Go!.
"

# ╔═╡ 217b9450-1f8b-11eb-26c1-1970f33d3543
begin
	@time x, stats = quick_algo_quad_lin_pbs(nlp)
	
	sol = x[1:nlp.meta.nvar]
	
	stats.status
end

# ╔═╡ a9e20bbc-1f8b-11eb-3b48-f39aff3b62ac
md"
**Step 6** We write the result in vtk files and visualize with Paraview.
"

# ╔═╡ b72a480c-1f8b-11eb-3ed3-1b239eaa82a4
begin

	yu = FEFunction(nlp.Y, sol)
	y, u = yu

	writevtk(trian,"data/results-u",cellfields=["uh"=>u])
	writevtk(trian,"data/results-y",cellfields=["yh"=>y])
	
	#The end.
end

# ╔═╡ 1e10021c-1f77-11eb-33fa-59697abc3514
md"## Extensions

* This also works for non-linear PDEs (with efficient AD).
* Handle pointwise constraints (for instance bound constraints).
* In our problem formulation, we might be interested in a **third type of variables** which are **discrete**. For instance, `` c \in \mathbb{R}_+`` is an unknown parameter of the model, and we consider
```math
\begin{equation}
	\begin{array}{clc}
	\min\limits_{y(x) \in \mathbb{R},u(x) \in \mathbb{R},c \in \mathbb{R}_+ } \  & \int_{\Omega} |y - y_d(x)|^2 + \alpha|u|^2dx + l(c), \\
	\mbox{ s.t. } & \int_{\Omega}c \nabla y ⋅ \nabla vdx = \int_{\Omega}(w + u)vdx, &\quad \text{ in } \Omega, \forall v \in V,\\
                  & y = 0, &\quad \text{ on } \partial \Omega.
	\end{array}
	\end{equation}
```
* Not tested yet is evolution PDEs, i.e. with a time-derivative.
* Now all the matrix-vector products are done by computing the (sparse) matrix first.
"

# ╔═╡ 6894eb7c-1f07-11eb-0125-39ed0c8d4b53
md"## References

* Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia. Journal of Open Source Software, 5(52), 2520.
* [Gridap.jl tutorial](https://gridap.github.io/Tutorials/stable/) 

"

# ╔═╡ Cell order:
# ╠═1b2edd22-1f82-11eb-26aa-b9db6cb28123
# ╟─a5ff4dbe-1f06-11eb-39a6-cbd2ca56ade2
# ╟─f3b870ce-1f08-11eb-298e-df588db76ef4
# ╟─b8499954-1f07-11eb-20d1-9f96b605a016
# ╟─91c7e358-1f09-11eb-2d64-eb8adb028644
# ╟─965fc458-1f13-11eb-0ecf-53154f37c23a
# ╠═31952da6-1f0e-11eb-2266-85a97813c35c
# ╟─1e0b77d8-1f84-11eb-2a68-77d7c0297260
# ╠═dddbaad2-1f82-11eb-1b16-19dc9fcfe111
# ╟─e7ed13fe-1f84-11eb-0101-3dbe71d41f09
# ╠═d840767c-1f83-11eb-006b-3f726d9a5168
# ╟─82d9a9ea-1f85-11eb-010a-a7a170ee5ab9
# ╠═f2e34852-1f82-11eb-0fec-4b84dcc82272
# ╟─4fd4f7ec-1f86-11eb-1553-cdf04b28889c
# ╠═5d72f79e-1f86-11eb-12ba-c3546a5bbed9
# ╟─929926b0-1f0f-11eb-3fa0-8d5815006d72
# ╠═d55fcdfa-1f0f-11eb-3bec-c7af8c6a7cee
# ╟─1fbad738-1f87-11eb-1dd3-cb7cbd083b74
# ╠═e520d750-1f11-11eb-2be9-bdd6b1e73125
# ╟─156e53ec-1f12-11eb-2766-8f32aadc1b9e
# ╟─7bd0918c-1f89-11eb-022d-3753157b3a34
# ╠═08451c3e-1f12-11eb-23c2-e93f32e5980e
# ╟─a6cc78ba-1f89-11eb-1c7c-4b6d8e45529b
# ╠═c2fdf630-1f89-11eb-3ede-1d2e96a49efb
# ╟─b24cf81c-1f8a-11eb-339c-4139a5064049
# ╠═3360adfc-1f8a-11eb-3a6e-77dfa01e929d
# ╠═55fbe488-1f99-11eb-0c13-97a3126e320b
# ╟─a11246d0-1f8b-11eb-1273-a795fc9802ab
# ╠═217b9450-1f8b-11eb-26c1-1970f33d3543
# ╟─a9e20bbc-1f8b-11eb-3b48-f39aff3b62ac
# ╠═b72a480c-1f8b-11eb-3ed3-1b239eaa82a4
# ╟─1e10021c-1f77-11eb-33fa-59697abc3514
# ╟─6894eb7c-1f07-11eb-0125-39ed0c8d4b53
