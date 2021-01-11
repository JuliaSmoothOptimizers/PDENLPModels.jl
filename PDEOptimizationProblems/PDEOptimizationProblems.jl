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
#                    - the parameter n should reflect the size of the problem
#                    - keep making test problems out of the Gridap tutorials:
#                    remains 3, 4, 5, 6, 7, 9, 10, 11
#                    https://gridap.github.io/Tutorials/stable/pages/t003_elasticity/
#                    - take the models from Gridap.jl instead of copy-paste.
#https://github.com/gridap/Gridap.jl/blob/master/test/GridapTests/PoissonTests.jl
#https://github.com/gridap/Gridap.jl/blob/master/test/GridapTests/StokesTaylorHoodTests.jl
#
#Maybe this is better, but I prefer having comments on each link (for now)
#=
path = dirname(@__FILE__)
files = filter(x -> x[end-2:end] == ".jl", readdir(path))
for file in files
  if file ≠ "OptimizationProblems.jl"
    include(file)
  end
end
=#
#end
###############################################################################
module PDEOptimizationProblems

using Gridap

using PDENLPModels

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
include("Burger1d.jl")
include("poissonBoltzman2d.jl")
include("smallestLaplacianeigenvalue.jl")
include("poisson3d.jl")
include("inversePoissonproblem2d.jl") #to be completed (in particular target function + check other things)

end #end of module
