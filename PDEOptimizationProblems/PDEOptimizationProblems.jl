###############################################################################
#
# This folder contains a list of PDE-constrained optimization problems modeled
# with Gridap and PDENLPModels.
#
#https://github.com/gridap/Gridap.jl
#Cite: Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia. Journal of Open Source Software, 5(52), 2520.
#
# TODO/Improvements: - generate the inclusions of the files;
#                    - make this folder an independant module.
#
#path = dirname(@__FILE__)
#files = filter(x->x[end-2:end] == ".jl", readdir(path))
#for file in files
#  if (file == "PDEOptimizationProblems.jl"); continue; end
#  include(file)
#end
###############################################################################
module PDEOptimizationProblems

using Gridap

using Main.PDENLPModels

#Unconstrained problems
include("penalizedPoisson.jl")

#PDE-constraint only
include("incompressibleNavierStokes.jl")

#Affine constraints
include("poissonmixte.jl")
#Affine constraints + control bounds
include("controlelasticmembrane1.jl") #constant bounds
include("controlelasticmembrane2.jl") #bounds applied to midcells

#Nonlinear constraints
include("1d-Burger.jl")
include("poissonBoltzman2d.jl")
include("smallestLaplacianeigenvalue.jl")
include("inversePoissonproblem2d.jl") #to be completed (in particular target function + check other things)

end #end of module
