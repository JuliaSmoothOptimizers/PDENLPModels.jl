### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ 31952da6-1f0e-11eb-2266-85a97813c35c
begin
	using Gridap

	#First, we describe the domain Ω
	n = 20
    domain = (0,1)
    partition = n
    model = CartesianDiscreteModel(domain,partition)

	#and define the machinery to compute integrals
	trian = Triangulation(model) #integration mesh
	degree = 1
	quad = CellQuadrature(trian,degree)	#quadrature
end

# ╔═╡ 156e53ec-1f12-11eb-2766-8f32aadc1b9e
begin

	using Main.workspace3.PDENLPModels
end

# ╔═╡ 13b301fe-250e-11eb-18b3-aff68783376c
begin

	using FastClosures, Krylov, LinearAlgebra, LinearOperators, NLPModels, Krylov, Main.workspace3.SQPFactFree
	
end

# ╔═╡ 1b2edd22-1f82-11eb-26aa-b9db6cb28123
begin
	
	include("../src/PDENLPModels.jl")

end

# ╔═╡ c2fdf630-1f89-11eb-3ede-1d2e96a49efb
begin
	
	include("../SQPalgorithm/SQP-factorization-free.jl")
	
end

# ╔═╡ a5ff4dbe-1f06-11eb-39a6-cbd2ca56ade2
md"## How to Use GridapPDENLPModels II: the nonlinear case
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


# ╔═╡ 400b0f16-250a-11eb-3ad3-7fee8c81bbc6


# ╔═╡ 965fc458-1f13-11eb-0ecf-53154f37c23a
md"
### Solve a nonlinear PDE with Gridap
For a fixed control, let us see how we can use Gridap to solve the PDE. "

# ╔═╡ 427f1846-250a-11eb-1e54-854a6e23bf65
md"
For instance, consider the stationary Burgers' equation:
```math
\begin{equation}
	\begin{array}{clc}
	& -\nu \frac{\partial^2 y}{\partial x^2} + y \frac{\partial y}{\partial x} = h(x) + u(x), &\quad x \in \Omega:=(0,1),\\
                  & y(0) = 0, y(1)=-1.
	\end{array}
	\end{equation}
```
where
```math
h(x) = 2(\nu + x^3), \text{ and } u = 0.5
```
"

# ╔═╡ 45d4a2fc-250a-11eb-0eb9-631c76f95c73
md"**Step 1** we describe the discretized domain Ω and the machinery to compute integrals."

# ╔═╡ 1e0b77d8-1f84-11eb-2a68-77d7c0297260
md"
**Step 2** Then, we describe the spaces where the variables are living. There are two types of functions: *test functions*, and the solution (in Gridap called *trial functions*). At this stage, we also specify the Dirichlet boundary conditions.
"

# ╔═╡ dddbaad2-1f82-11eb-1b16-19dc9fcfe111
begin

	labels = get_face_labeling(model)
	add_tag_from_tags!(labels,"diri1",[2])
	add_tag_from_tags!(labels,"diri0",[1])
	
	#Then, we describe the spaces where the variables are living. There are two types of functions. Test functions:
	V = TestFESpace(
	  reffe=:Lagrangian, 
	  conformity=:H1, #this space is a subset of H^1(Ω)
	  valuetype=Float64, #the test and trial functions are real-valued
	  model=model, 
	  labels=labels,
	  order=1, #first order, Lagrangian interpolation of the function at FE element
	  dirichlet_tags=["diri0", "diri1"]) #tags refer to a part/region of the domain

	#and the "solution" function:
	uD0 = VectorValue(0)
	uD1 = VectorValue(-1)
	U   = TrialFESpace(V, [uD0, uD1])
	
end

# ╔═╡ e7ed13fe-1f84-11eb-0101-3dbe71d41f09
md"
**Step 3** Once the framework is set, we define the problem. Each integral is described with a **FETerm**. In our example, we have
``
$res(y,v)= \int_{\Omega}-\nu\frac{\partial y}{\partial x} \frac{\partial v}{\partial x} + y \frac{\partial y}{\partial x} v - hv - u v~dx.$
``
Note here that we could have split this term in a linear and a nonlinear terms.
"

# ╔═╡ d840767c-1f83-11eb-006b-3f726d9a5168
begin

	#Now, we describe the equation we want to solve.
	#Each integral is described with a FETerm.
	@law conv(y,∇y) = (∇y ⋅one(∇y))⊙y

	nu = 0.08
	h(x) = 2*(nu + x[1]^3)
	function res_pde(y,v)
	  #u(x) = 0.5
	  -nu*(∇(v)⊙∇(y)) + v⊙conv(y,∇(y)) - v * 0.5 - v * h
	end
	t_Ω = FETerm(res_pde, trian, quad)

	#Terms are aggregated in a FEOperator:
	op_pde = FEOperator(U,V,t_Ω)
	
end

# ╔═╡ f2e34852-1f82-11eb-0fec-4b84dcc82272
begin

	#Gridap's suggestion to solve the nonlinear problem:	
	using LineSearches: BackTracking
	nls = NLSolver(
	  show_trace=true, method=:newton, linesearch=BackTracking())
	solver = FESolver(nls)

	uh = solve(solver,op_pde)
	#convert the FEFunction uh into a Vector.
	sol_gridap = get_free_values(uh)
	
	#We can print the residual:
	@show norm(Gridap.FESpaces.residual(op_pde, uh), Inf)
	
end

# ╔═╡ 82d9a9ea-1f85-11eb-010a-a7a170ee5ab9
md"
**Step 4** The problem is now reduced to a nonlinear equation that we can solve with our favorite approach.
"

# ╔═╡ 4fd4f7ec-1f86-11eb-1553-cdf04b28889c
md"
**Step 5** Visualize the solution using Paraview (not in Julia for now).
"

# ╔═╡ 5d72f79e-1f86-11eb-12ba-c3546a5bbed9
begin

	#and print the solution (using Paraview):
	writevtk(trian,"data/results_nl",cellfields=["uh"=>uh])
	
end

# ╔═╡ 929926b0-1f0f-11eb-3fa0-8d5815006d72
md" 
### Solve a (nonlinear) PDE-constrained opimization problem with Gridap and NLPModels

For instance, consider the stationary Burgers' equation:
```math
\begin{equation}
	\begin{array}{clc}
	\min\limits_{y(x) \in \mathbb{R},u(x) \in \mathbb{R} } \  & \int_{\Omega} |y - y_d(x)|^2 + \alpha|u-0.5|^2dx, \\
	\mbox{ s.t. } & -\nu \frac{\partial^2 y}{\partial x^2} + y \frac{\partial y}{\partial x} = h(x) + u(x), &\quad x \in \Omega:=(0,1),\\
                  & y(0) = 0, y(1)=-1.
	\end{array}
	\end{equation}
```
where
```math
h(x) = 2(\nu + x^3), \text{ and } u = 0.5
```

We can proceed in a similar way, to define our GridapPDENLPModel, by:
* Setting the Test and Trial spaces;
* Model the constraints with an FEOperator;
* Model the objective function in a similar way.

**Step 1** We describe the discretized domain Ω and the machinery to compute integrals. We reuse the one defined before.

**Step 2** We define the test and trial spaces. Note that for the control, there is a no Dirichlet boundary conditions.
" 

# ╔═╡ d55fcdfa-1f0f-11eb-3bec-c7af8c6a7cee
begin
	
	Xedp = V
	Yedp = U
	Xcon = TestFESpace(
			reffe=:Lagrangian, order=1, valuetype=Float64,
			conformity=:H1, model=model)
	Ycon = TrialFESpace(Xcon)
	
	#It will be convenient to have the whole solution+control space.
	Y = MultiFieldFESpace([Yedp, Ycon])
	
	#We use Gridap.FESpaces.num_free_dofs to get the size of the variables.
	#Here Yedp is more constraint due to the Dirichlet conditions.
	Gridap.FESpaces.num_free_dofs(Yedp) < Gridap.FESpaces.num_free_dofs(Ycon)

end

# ╔═╡ 1fbad738-1f87-11eb-1dd3-cb7cbd083b74
md"
**Step 3** Once the framework is set, we define the problem. Each integral is described with a **FETerm**. In our example, we have
``
$res(y,u,v)= \int_{\Omega}-\nu\frac{\partial y}{\partial x} \frac{\partial v}{\partial x} + y \frac{\partial y}{\partial x} v - hv - u v~dx.$
``
and
``
$f(y,u) = \int_{\Omega} |y - y_d(x)|^2 + 100 |u - 0.5|^2dx$
``
We have only one type of integral here, so one FETerm.
"

# ╔═╡ e520d750-1f11-11eb-2be9-bdd6b1e73125
begin

	yd(x) = -x[1]^2
	α = 1e2
	#objective function:
	f(y, u) = 0.5 * (yd - y) * (yd - y) + 0.5 * α * (u - 0.5) * (u - 0.5)
	function f(yu)
		y, u = yu
		f(y,u)
	end

	function res(yu, v) #u is the solution of the PDE and z the control
	 y, u = yu
	 v

	 -nu*(∇(v)⊙∇(y)) + v⊙conv(y,∇(y)) - v * u - v * h
	end
	term = FETerm(res,trian,quad)
	op = FEOperator(Y, V, term)

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

# ╔═╡ 3360adfc-1f8a-11eb-3a6e-77dfa01e929d
begin
function solve_saddle_system(nlp :: AbstractNLPModel;
                              x0  :: AbstractVector{T} = nlp.meta.x0,
                              itmax :: Int = 100) where T

  #First step, we form the system
  # ( G  A' ) (x) = (-g)
  # ( A  0  ) (λ) = ( b)
  #G = hess(nlp, x0)
  #A = jac(nlp,  x0)
  #g = grad(nlp, 0.)
  #b = cons(nlp, 0.)

  rhs = vcat(- grad(nlp, zeros(nlp.meta.nvar)), cons(nlp, zeros(nlp.meta.nvar)))

  #We now create a LinearOperator
  jacop = jac_op(nlp, x0)
  Jv  = Array{T,1}(undef, nlp.meta.ncon)
  Jtv = Array{T,1}(undef, nlp.meta.nvar)
  prod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jacop'*v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon],
                            jacop*v[1:nlp.meta.nvar])
  ctprod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jacop'*v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon],
                            jacop*v[1:nlp.meta.nvar])
 #PreallocatedLinearOperator{T}(nrow::Int, ncol::Int, symmetric::Bool, hermitian::Bool, prod, tprod, ctprod)
  sad_op = PreallocatedLinearOperator{T}(nlp.meta.ncon+nlp.meta.nvar, nlp.meta.ncon+nlp.meta.nvar, true, true, prod, ctprod, ctprod)

  #(x, stats) = symmlq(sad_op, rhs)
  (x, stats) = minres(sad_op, rhs, itmax = itmax)

  return x, stats
 end
end

# ╔═╡ 46b20212-250e-11eb-03a8-158c61f50b1e
begin
	function sqp_solver2(nlp      :: AbstractNLPModel;
                         x0       :: AbstractVector{T} = nlp.meta.x0,
                         atol     :: AbstractFloat = 1e-3,
                         max_iter :: Int = 10,
                         itmax    :: Int = 100) where T

  x = copy(x0)
  lqp = SQPNLP(nlp, copy(x0), zeros(nlp.meta.ncon))
  score = norm(lqp.gx, Inf)
  OK = score <= atol

  #h = LineModel(nlp, x, lqp.gx)

  #@info log_header([:iter, :f, :c, :score, :sigma], [Int, T, T, T, T],
   #                hdr_override=Dict(:f=>"f(x)", :c=>"||c(x)||", :score=>"‖∇L‖", :sigma=>"σ"))
  #@info log_row(Any[0, lqp.fx, norm(lqp.cx,Inf), score, lqp.delta])

  i=0
  while !OK
   p, stats = solve_saddle_system(lqp, x0 = zeros(nlp.meta.nvar), itmax = itmax)
   if ~stats.solved @show stats.status end

   #redirect!(h, vcat(lqp.xk,lqp.lk), p)
   #slope = dot(nlp.meta.nvar, p, lqp.gx)
   # Perform improved Armijo linesearch.
   #t, good_grad, ft, nbk, nbW = armijo_wolfe(h, f, slope, ∇ft, τ₁=T(0.9999), bk_max=25, verbose=false)
   #t, good_grad, ft, nbk, nbW = lsfunc(h, lqp.fx, slope, lqp.gx, τ₁=T(0.9999), bk_max=25, verbose=false)

   lqp.xk += p[1:nlp.meta.nvar]
   lqp.lk += p[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon]

   i += 1
   score = norm(p[1:nlp.meta.nvar], Inf)
   OK = (score <= atol) || (i > max_iter)
   if !OK
    lqp.fx, lqp.gx, lqp.cx = obj(nlp, lqp.xk), grad(nlp, lqp.xk), cons(nlp, lqp.xk)
   end
   #@info log_row(Any[i, lqp.fx, norm(lqp.cx,Inf), norm(score,Inf), lqp.delta])
  end #end of main loop

  #stats = GenericExecutionStats((score <= atol), nlp, solution = lqp.xk)

  return lqp.xk
 end
end

# ╔═╡ a11246d0-1f8b-11eb-1273-a795fc9802ab
md"
**Step 5** Go!.
"

# ╔═╡ 217b9450-1f8b-11eb-26c1-1970f33d3543
begin
	#Great initial guess
	x0 = vcat(sol_gridap, 0.5*ones(Gridap.FESpaces.num_free_dofs(Ycon)))

	@time x = sqp_solver2(nlp, x0 = x0, atol = 1e-3, max_iter = 10, itmax = 1000)

	obj(nlp, x) < obj(nlp, x0)

end

# ╔═╡ d9b62480-2518-11eb-3660-6b090ea65c25
begin
	norm(cons(nlp, x),Inf)
end

# ╔═╡ a9e20bbc-1f8b-11eb-3b48-f39aff3b62ac
md"
**Step 6** We write the result in vtk files and visualize with Paraview.
"

# ╔═╡ b72a480c-1f8b-11eb-3ed3-1b239eaa82a4
begin

	yu = FEFunction(nlp.Y, x)
	y, u = yu

	writevtk(trian,"data/results-nl-u",cellfields=["uh"=>u])
	writevtk(trian,"data/results-nl-y",cellfields=["yh"=>y])
	
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
# ╟─1b2edd22-1f82-11eb-26aa-b9db6cb28123
# ╟─a5ff4dbe-1f06-11eb-39a6-cbd2ca56ade2
# ╟─f3b870ce-1f08-11eb-298e-df588db76ef4
# ╟─b8499954-1f07-11eb-20d1-9f96b605a016
# ╟─91c7e358-1f09-11eb-2d64-eb8adb028644
# ╠═400b0f16-250a-11eb-3ad3-7fee8c81bbc6
# ╟─965fc458-1f13-11eb-0ecf-53154f37c23a
# ╟─427f1846-250a-11eb-1e54-854a6e23bf65
# ╟─45d4a2fc-250a-11eb-0eb9-631c76f95c73
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
# ╟─c2fdf630-1f89-11eb-3ede-1d2e96a49efb
# ╠═13b301fe-250e-11eb-18b3-aff68783376c
# ╠═3360adfc-1f8a-11eb-3a6e-77dfa01e929d
# ╠═46b20212-250e-11eb-03a8-158c61f50b1e
# ╟─a11246d0-1f8b-11eb-1273-a795fc9802ab
# ╠═217b9450-1f8b-11eb-26c1-1970f33d3543
# ╠═d9b62480-2518-11eb-3660-6b090ea65c25
# ╟─a9e20bbc-1f8b-11eb-3b48-f39aff3b62ac
# ╠═b72a480c-1f8b-11eb-3ed3-1b239eaa82a4
# ╟─1e10021c-1f77-11eb-33fa-59697abc3514
# ╟─6894eb7c-1f07-11eb-0125-39ed0c8d4b53
