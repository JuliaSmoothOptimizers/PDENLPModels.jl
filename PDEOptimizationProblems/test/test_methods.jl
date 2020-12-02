using Main.PDEOptimizationProblems

using Gridap, PDENLPModels, Test
using LinearAlgebra, SparseArrays, NLPModels

nlp = controlelasticmembrane1()

hessian_test_functions(nlp)
#=
Tanj, December 1st 2020
@btime hessian_test_functions(nlp)
  208.833 ms (654095 allocations: 237.33 MiB)
=#
