using Gridap
using BenchmarkTools, LinearAlgebra, NLPModels, SparseArrays, Test

using PDENLPModels

# Using Gridap and GridapPDENLPModel, we solve the following
# distributed Poisson control proble with Dirichlet boundary:
#
# min_{u,z}   0.5 ∫_Ω​ |u(x) - ud(x)|^2dx + 0.5 * α * ∫_Ω​ |z|^2
# s.t.         -Δu + sinh(u) = h + z,   for    x ∈  Ω
#                         u(x) = 0,     for    x ∈ ∂Ω

nothing;
