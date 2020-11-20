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
  if !ns.stats.solved @show ns.stats.status end
  x .= y
  x
end
