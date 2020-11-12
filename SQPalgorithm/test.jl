using Test

include("SQP-factorization-free.jl")
using Main.SQPFactFree

using OptimizationProblems

_model = hs111()

using NLPModelsJuMP

nlp = MathOptNLPModel(_model)
n, x0 = nlp.meta.nvar, nlp.meta.x0

x, stats = sqp_solver(nlp)
