using Gridap
using BenchmarkTools, LinearAlgebra, NLPModels, SparseArrays, Test

using PDENLPModels

#Exercice 10.2.11 (p. 313) in Allaire, Analyse numérique et optimisation, Les éditions de Polytechnique
#More eigenvalue problems can be found in Section 7.3.2
#
#
# We solve the following problem:
#
# min_{u,z}   ∫_Ω​ |∇u|^2
# s.t.        ∫_Ω​ u^2 = 1,     for    x ∈  Ω
#                u    = 0,     for    x ∈ ∂Ω
#
# The solution is an eigenvector of the smallest eigenvalue of the Laplacian operator,
# given by the value of the objective function.
# λ is an eigenvalue of the Laplacian if there exists u such that
#
# Δu + λ u = 0,   for    x ∈  Ω
#        u = 0,   for    x ∈ ∂Ω

nothing;
