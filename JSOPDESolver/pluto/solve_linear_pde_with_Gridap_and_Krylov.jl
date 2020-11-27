### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ ab184a36-2a86-11eb-209a-61c953a66cd9
begin
	using FastClosures, Krylov, LinearAlgebra, LinearOperators, SparseArrays
end

# ╔═╡ 8c6a965a-2a79-11eb-1328-1f6bc1386dab
begin
	using Gridap

	###############################################################################
	#Gridap resolution:
	#This corresponds to a Poisson equation with Dirichlet and Neumann conditions
	#described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/
	function _poisson()
		domain = (0,1,0,1)
		n = 8#2^7
		partition = (n,n)
		model = CartesianDiscreteModel(domain,partition)

		trian = Triangulation(model)
		degree = 2
		quad = CellQuadrature(trian,degree)

		V0 = TestFESpace(
		  reffe=:Lagrangian, order=1, valuetype=Float64,
		  conformity=:H1, model=model, dirichlet_tags="boundary")

		g(x) = 0.0
		Ug = TrialFESpace(V0,g)

		w(x) = 1.0
		a(u,v) = ∇(v)⊙∇(u)
		b_Ω(v) = v*w
		t_Ω = AffineFETerm(a,b_Ω,trian,quad)

		op_pde = AffineFEOperator(Ug,V0,t_Ω)
		return op_pde
	end

	op_pde = _poisson()
end

# ╔═╡ 94d6ae36-2a78-11eb-227c-5fb11d74248c
md"
## Use Krylov.jl to solve linear PDE

We mentioned last weeks, how to use Gridap to solve a PDE (no conrol/no optimization) using Gridap. Let us consider the Poisson equation
```math
\begin{equation}
	\begin{array}{clc}
	& -\Delta y(x) = g(x), &\quad x \in \Omega,\\
                  & y(x) = 0, &\quad x \in \partial \Omega.
	\end{array}
	\end{equation}
```
"

# ╔═╡ cc5b3686-2a86-11eb-04bd-0f89fa4c2994
md"
the size of the matrix is:
"

# ╔═╡ 6b4a2c38-2a79-11eb-361a-6da5af5148b9
begin
	size(get_matrix(op_pde))
end

# ╔═╡ c4796c78-2a86-11eb-29a1-19acf8503c5d
md"
It is a sparse matrix:
"

# ╔═╡ 9ddb208e-2a86-11eb-372b-d1bc1f5bc705
begin
	issparse(get_matrix(op_pde))
end

# ╔═╡ d4be6f0a-2a86-11eb-3629-03882b62a23d
md"
It is symmetric:
"

# ╔═╡ bf6dec84-2a86-11eb-2b70-7f07ddbb3908
begin
	issymmetric(get_matrix(op_pde))
end

# ╔═╡ ee720e16-2a86-11eb-3c7a-6956b274ddd5
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
	  x .= y
	  x
	end
end

# ╔═╡ 0af4dd5a-2a87-11eb-2e12-4b4d3cdd20cd
md"
The method suggested in Gridap is to initialize a *LinearSolver* and then use it to initialize a *LinearFESolver* that contains all the information needed by a *solve!* function.
"

# ╔═╡ faf6a26e-2a86-11eb-24ee-0529c59bed64
begin
	#Gridap.jl/src/FESpaces/FESolvers.jl
	#Gridap.jl/src/Algebra/LinearSolvers.jl
	ls  = KrylovSolver(cg; itmax = 15)
	ls1 = LUSolver()
	ls2 = BackslashSolver()

	solver  = LinearFESolver(ls)
	solver1 = LinearFESolver(ls1)
	solver2 = LinearFESolver(ls2)

	uh  = solve(solver, op_pde)
	x = get_free_values(uh) #The result is an FEFunction, so we convert it in a Vector
	uh1 = solve(solver1,op_pde)
	x1 = get_free_values(uh1)
	uh2 = solve(solver2,op_pde)
	x2 = get_free_values(uh2)

	@show norm(get_matrix(op_pde)*x  - get_vector(op_pde),Inf), norm(get_matrix(op_pde)*x1 - get_vector(op_pde),Inf), norm(get_matrix(op_pde)*x2 - get_vector(op_pde),Inf)
end

# ╔═╡ 50d32052-2a87-11eb-3d29-bd2ce3ff9bbd
md"
Running a very basic benchmark with @time for a problem with n=2^7, i.e. an (16129, 16129) matrix, gives:
  - 0.065231 seconds (182 allocations: 906.219 KiB) #minres
  - 0.139934 seconds (201 allocations: 29.457 MiB) #LU
  - 0.068973 seconds (182 allocations: 19.004 MiB) #\
which seems very positive!
"

# ╔═╡ 41841d42-2a87-11eb-02b5-1d9aeed0fd96
md"
This way was just surcharging the methods used by Gridap.jl. However, no output from the system is anticipated (yet).
"

# ╔═╡ 6250840c-2a8c-11eb-11c0-696ed5d0a545


# ╔═╡ Cell order:
# ╠═ab184a36-2a86-11eb-209a-61c953a66cd9
# ╟─94d6ae36-2a78-11eb-227c-5fb11d74248c
# ╠═8c6a965a-2a79-11eb-1328-1f6bc1386dab
# ╟─cc5b3686-2a86-11eb-04bd-0f89fa4c2994
# ╠═6b4a2c38-2a79-11eb-361a-6da5af5148b9
# ╟─c4796c78-2a86-11eb-29a1-19acf8503c5d
# ╠═9ddb208e-2a86-11eb-372b-d1bc1f5bc705
# ╟─d4be6f0a-2a86-11eb-3629-03882b62a23d
# ╠═bf6dec84-2a86-11eb-2b70-7f07ddbb3908
# ╠═ee720e16-2a86-11eb-3c7a-6956b274ddd5
# ╟─0af4dd5a-2a87-11eb-2e12-4b4d3cdd20cd
# ╠═faf6a26e-2a86-11eb-24ee-0529c59bed64
# ╟─50d32052-2a87-11eb-3d29-bd2ce3ff9bbd
# ╟─41841d42-2a87-11eb-02b5-1d9aeed0fd96
# ╟─6250840c-2a8c-11eb-11c0-696ed5d0a545
