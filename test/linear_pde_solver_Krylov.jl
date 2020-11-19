using FastClosures, Gridap, Krylov, LinearAlgebra, LinearOperators, NLPModels, SparseArrays
using Test, BenchmarkTools

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


###############################################################################
#Gridap resolution:
#This corresponds to a Poisson equation with Dirichlet and Neumann conditions
#described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/
function _poisson()
    domain = (0,1,0,1)
    n = 2^7
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

#Gridap.jl/src/FESpaces/FESolvers.jl
#Gridap.jl/src/Algebra/LinearSolvers.jl
@time ls  = KrylovSolver(minres; itmax = 150)
@time ls1 = LUSolver()
@time ls2 = BackslashSolver()

solver  = LinearFESolver(ls)
solver1 = LinearFESolver(ls1)
solver2 = LinearFESolver(ls2)

#Describe the matrix:
@test size(get_matrix(op_pde)) == (16129, 16129)
@test issparse(get_matrix(op_pde))
@test issymmetric(get_matrix(op_pde))

uh  = solve(solver, op_pde)
uh1 = solve(solver1,op_pde)
uh2 = solve(solver2,op_pde)
#Sad, that we don't have the stats back...

@time uh = solve(solver,op_pde)
x = get_free_values(uh)
@time uh1 = solve(solver1,op_pde)
x1 = get_free_values(uh1)
@time uh2 = solve(solver2,op_pde)
x2 = get_free_values(uh2)

@test norm(x  - x1, Inf) <= 1e-8
@test norm(x1 - x2, Inf) <= 1e-13
@show norm(get_matrix(op_pde)*x  - get_vector(op_pde),Inf) <= 1e-8
@test norm(get_matrix(op_pde)*x1 - get_vector(op_pde),Inf) <= 1e-15
@test norm(get_matrix(op_pde)*x2 - get_vector(op_pde),Inf) <= 1e-15
