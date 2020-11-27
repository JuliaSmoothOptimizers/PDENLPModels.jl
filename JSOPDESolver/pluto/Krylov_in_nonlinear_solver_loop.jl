### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ f09383d6-3002-11eb-3732-f580e5aea9a0
begin
	using Gridap, PDENLPModels
	using FastClosures,Krylov, LinearAlgebra, LinearOperators, NLPModels, SparseArrays
	using Test, BenchmarkTools
	using NLPModelsIpopt
end

# ╔═╡ 6c2d5bfe-3003-11eb-09d2-b9bd200d0a95
md"
### Solve nonlinear PDEs with Krylov.jl

Using the framework proposed in Gridap, we can use Krylov.jl functions to solve the linear system in a Newton loop.
The alternative would be to surcharge the `solve!` function:

`solve!(x::AbstractVector,nls::NewNonlinearSolverType,op::NonlinearOperator,cache)`

First, we include the KrylovLinearSolver structure:
"

# ╔═╡ 60403a10-3004-11eb-0f8c-e55efc86c99e
begin
	struct KrylovSolver <: Gridap.Algebra.LinearSolver
	 krylov_func :: Function
	 kwargs      :: Dict
	end

	function KrylovSolver(krylov_func :: Function;kwargs...)
	 return KrylovSolver(krylov_func, kwargs)
	end

	struct KrylovSymbolicSetup <: Gridap.Algebra.SymbolicSetup
		krylov_func :: Function
		kwargs      :: Dict
	end

	mutable struct KrylovNumericalSetup{LO,T} <: Gridap.Algebra.NumericalSetup

		linear_op   :: LO

		krylov_func :: Function
		stats       :: Union{Krylov.KrylovStats{T},Nothing}
		kwargs      :: Dict

		function KrylovNumericalSetup(T        :: Type,
									  A        :: LO,
									  krylov_func :: Function,
									  kwargs   :: Dict;
									  stats    = nothing,#:: Union{Krylov.KrylovStats{T},Nothing} = nothing,
									  ) where LO

		  return new{LO,T}(A, krylov_func, stats, kwargs)
		end
	end

	import Gridap.Algebra: symbolic_setup, numerical_setup, numerical_setup!, solve!
	symbolic_setup(solver::KrylovSolver,mat::AbstractMatrix) = KrylovSymbolicSetup(solver.krylov_func, solver.kwargs)

	function numerical_setup(kss::KrylovSymbolicSetup, mat::AbstractMatrix{T}) where T

		#m, n = size(mat)
		#Jv  = Array{T,1}(undef, m)
		#Jtv = Array{T,1}(undef, n)
		#prod = @closure v ->  mat*v
		#ctprod = @closure v ->  mat'*v

		#op = PreallocatedLinearOperator{T}(m, n, false, true, prod, ctprod, ctprod)

		#KrylovNumericalSetup(T, op, kss.krylov_func, kss.kwargs)
		KrylovNumericalSetup(T, mat, kss.krylov_func, kss.kwargs)
	end

	function numerical_setup!(ns::KrylovNumericalSetup, mat::AbstractMatrix)
	nothing #apparently don't pass by here
	end

	function solve!(x::AbstractVector,ns::KrylovNumericalSetup,b::AbstractVector)
	  (y, ns.stats) = ns.krylov_func(ns.linear_op, b; ns.kwargs...)
	  if !ns.stats.solved @show ns.stats.status end
	  x .= y
	  x
	end
end

# ╔═╡ 5a8108e0-3005-11eb-15fb-1980104775b9
begin
	#Gridap way of solving the equation:
	using LineSearches
	nls = NLSolver(
	  show_trace=false, method=:newton, linesearch=BackTracking())
	solver = FESolver(nls)

	#A non-documented alternative:
	#struct NewtonRaphsonSolver <:NonlinearSolver
	#  ls::LinearSolver
	#  tol::Float64
	#  max_nliters::Int
	#end
	nls2 = Gridap.Algebra.NewtonRaphsonSolver(LUSolver(), 1e-6, 100)
	solver2 = FESolver(nls2)

	#The first approach is to use Newton method anticipated by Gridap and using
	#Krylov.jl to solve the linear problem.
	#NLSolver(ls::LinearSolver;kwargs...)
	ls  = KrylovSolver(cgls; itmax = 1000, verbose = false)
	nls_krylov = NLSolver(ls, show_trace=false)
	@test nls_krylov.ls == ls
	solver_krylov = FESolver(nls_krylov)

	nls_krylov2 = Gridap.Algebra.NewtonRaphsonSolver(ls, 1e-6, 100)
	solver_krylov2 = FESolver(nls_krylov2)
end

# ╔═╡ 03de4a32-3005-11eb-3ab4-fb230180065f
md"
Then, we initialize a nonlinear PDE problem:
"

# ╔═╡ 1270c9a6-3004-11eb-0223-dbcc947014d8
begin
	function _pdeonlyincompressibleNS()
		n = 10
		domain = (0,1,0,1)
		partition = (n,n)
		model = CartesianDiscreteModel(domain,partition)

		labels = get_face_labeling(model)
		add_tag_from_tags!(labels,"diri1",[6,])
		add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

		D = 2
		order = 2
		V = TestFESpace(
		  reffe=:Lagrangian, conformity=:H1, valuetype=VectorValue{D,Float64},
		  model=model, labels=labels, order=order, dirichlet_tags=["diri0","diri1"])

		Q = TestFESpace(
		  reffe=:PLagrangian, conformity=:L2, valuetype=Float64,
		  model=model, order=order-1, constraint=:zeromean)

		uD0 = VectorValue(0,0)
		uD1 = VectorValue(1,0)
		U = TrialFESpace(V,[uD0,uD1])
		P = TrialFESpace(Q)

		X = MultiFieldFESpace([V, Q])
		Y = MultiFieldFESpace([U, P])

		Re = 10.0
		@law conv(u,∇u) = Re*(∇u')⋅u
		@law dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

		function a(y,x)
		  u, p = y
		  v, q = x
		  ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u)
		end

		c(u,v) = v⊙conv(u,∇(u))
		dc(u,du,v) = v⊙dconv(du,∇(du),u,∇(u))

		function res(y,x)
		  u, p = y
		  v, q = x
		  a(y,x) + c(u,v)
		end

		function ja(y,dy,x)
		  u, p = y
		  v, q = x
		  du, dp = dy
		  a(dy,x)+ dc(u,du,v)
		end

		trian = Triangulation(model)
		degree = (order-1)*2
		quad = CellQuadrature(trian,degree)
		t_Ω = FETerm(res,trian,quad)
		op = FEOperator(Y,X,t_Ω)

		t_with_jac_Ω = FETerm(res,ja,trian,quad)
		op_with_jac = FEOperator(Y,X,t_with_jac_Ω)
	end

	op = _pdeonlyincompressibleNS()

	num_free_dofs(op.trial) #number of variables
end

# ╔═╡ 47377e4c-3005-11eb-2717-fd5723643ef3
md"
Before solving, we also initialize the `FESolver`
"

# ╔═╡ 7cdc133a-3005-11eb-1b15-b31136b10334
md"
Finally, we can proceed and solve the problem.
"

# ╔═╡ 8978e1c2-3005-11eb-08ef-9ba926411b19
begin
	@time uph1 = solve(solver,op)
	sol_gridap1 = get_free_values(uph1);
	@time uph2 = solve(solver2,op)
	sol_gridap2 = get_free_values(uph2);
	@time uph3 = solve(solver_krylov,op)
	sol_gridap3 = get_free_values(uph3);
	#@time uph4 = solve(solver_krylov2,op)
	#sol_gridap4 = get_free_values(uph4);

	@show norm(Gridap.FESpaces.residual(op, uph1),Inf)
	@show norm(Gridap.FESpaces.residual(op, uph2),Inf)
	@show norm(Gridap.FESpaces.residual(op, uph3),Inf)
	#@show norm(Gridap.FESpaces.residual(op, uph4),Inf)
end

# ╔═╡ 99726372-3006-11eb-1b87-2dcd224a7e42
md"
Checking the performance, without fine tuning, it is probably not the best way (and there is clearly a memory issue).
"

# ╔═╡ Cell order:
# ╠═f09383d6-3002-11eb-3732-f580e5aea9a0
# ╟─1701cf3e-3003-11eb-0799-31bd18b58612
# ╟─2e8a731e-3007-11eb-0db3-1982d09655d6
# ╟─21cd491c-300c-11eb-1808-3302b22f498e
# ╠═0d89ffda-300d-11eb-19da-497d65a439ad
# ╟─4f519284-300d-11eb-3c14-2fd7a819c45c
# ╠═4c82a886-300d-11eb-177a-a70b223aec06
# ╟─76c7745a-300d-11eb-01cd-957197d61a3c
# ╠═7c8f53b2-300d-11eb-18f1-4f461c2fc0c3
# ╟─6c2d5bfe-3003-11eb-09d2-b9bd200d0a95
# ╟─60403a10-3004-11eb-0f8c-e55efc86c99e
# ╟─03de4a32-3005-11eb-3ab4-fb230180065f
# ╠═1270c9a6-3004-11eb-0223-dbcc947014d8
# ╟─47377e4c-3005-11eb-2717-fd5723643ef3
# ╠═5a8108e0-3005-11eb-15fb-1980104775b9
# ╟─7cdc133a-3005-11eb-1b15-b31136b10334
# ╠═8978e1c2-3005-11eb-08ef-9ba926411b19
# ╟─99726372-3006-11eb-1b87-2dcd224a7e42
