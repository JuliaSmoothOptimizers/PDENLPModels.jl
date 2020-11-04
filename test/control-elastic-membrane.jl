using Gridap
using BenchmarkTools, LinearAlgebra, NLPModels, Main.PDENLPModels, SparseArrays, Test

using Main.PDENLPModels

# Using Gridap and GridapPDENLPModel, we solve the following
# distributed Poisson control proble with Dirichlet boundary:
#
# min_{y,u}   0.5 ∫_Ω​ |y(x) - yd(x)|^2dx + 0.5 * α * ∫_Ω​ |u|^2
# s.t.         -Δy = h + u,   for    x ∈  Ω
#               y  = 0,       for    x ∈ ∂Ω
#              u_min(x) <=  u(x) <= u_max(x)

nothing;
