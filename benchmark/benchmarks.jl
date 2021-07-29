using DelimitedFiles, LinearAlgebra, Printf, SparseArrays
using BenchmarkTools, DataFrames, Plots
#JSO packages
using NLPModels, BenchmarkProfiles, SolverBenchmark
#This package
using Gridap, PDENLPModels, PDEOptimizationProblems

fun = Dict(
  :obj => (nlp, x) -> obj(nlp, x),
  :grad => (nlp, x) -> grad(nlp, x),
  :hess_coord => (nlp, x) -> hess_coord(nlp, x),
  :hess_structure => (nlp, x) -> hess_structure(nlp),
  :jac_coord => (nlp, x) -> (nlp.meta.ncon > 0 ? jac_coord(nlp, x) : zero(eltype(x))),
  :jac_structure => (nlp, x) -> (nlp.meta.ncon > 0 ? jac_structure(nlp) : zero(eltype(x))),
  :hess_lag_coord => (nlp, x) -> hess_coord(nlp, x, ones(nlp.meta.ncon)),
)
problems = PDEOptimizationProblems.problems[1:3]

const SUITE = BenchmarkGroup()
for f in keys(fun)
  SUITE[f] = BenchmarkGroup()
  for pb in problems
    SUITE[f][pb] = BenchmarkGroup()
  end
end

for pb in problems
  npb = eval(Meta.parse("PDEOptimizationProblems.$(pb)()")) # add a kwargs n=... to modify the size
  for (fs, f) in fun
    x = npb.meta.x0
    SUITE[fs][pb] = @benchmarkable eval($f)($npb, $x)
  end
end
