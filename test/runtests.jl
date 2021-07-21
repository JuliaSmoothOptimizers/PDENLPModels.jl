#This package
using Gridap, PDENLPModels
using PDENLPModels:
  FEFunctionType,
  _split_vector,
  _split_FEFunction,
  _obj_integral,
  _compute_gradient_k,
  _compute_gradient!
#Testing
using NLPModels, NLPModelsIpopt, NLPModelsTest, Random, Test
using LinearAlgebra, SparseArrays

Random.seed!(1998)

n = 3
#Tanj: for each problem there is a lowercase(problem) function that returns a GridapPDENLPModel
#++ would be to also have a lowercase(problem)_test that test the problem with the exact solution.
local_test = false # tested locally only

pde_problems = if local_test
  [
    "BURGER1D",
    "CELLINCREASE",
    "SIS",
    "CONTROLSIR",
    "DYNAMICSIR",
    "BASICUNCONSTRAINED",
    "PENALIZEDPOISSON",
    # "INCOMPRESSIBLENAVIERSTOKES", #too slow (tested locally only)
    "POISSONMIXED",
    "POISSONPARAM",
    "POISSONMIXED2",
    "TOREBRACHISTOCHRONE",
    "CONTROLELASTICMEMBRANE",
  ]
else
  [
    "BURGER1D",
    # "CELLINCREASE",
    # "SIS",
    "CONTROLSIR", # similar SIS, CELLINCREASE, DYNAMICSIR
    # "DYNAMICSIR",
    # "BASICUNCONSTRAINED", # simplified PENALIZEDPOISSON
    "PENALIZEDPOISSON",
    #"INCOMPRESSIBLENAVIERSTOKES", #too slow (tested locally only)
    "POISSONMIXED",
    "POISSONPARAM",
    # "POISSONMIXED2", # similar POISSONMIXED
    "TOREBRACHISTOCHRONE",
    "CONTROLELASTICMEMBRANE",
  ]
end

for problem in pde_problems
  include("problems/$(lowercase(problem)).jl")
end

@testset "NLP tests" begin
  for problem in pde_problems
    @info "$(problem)"
    @time nlp = eval(Meta.parse("$(lowercase(problem))(n=$(n))"))
    @testset "Test problem scenario" begin
      if local_test
        @time eval(Meta.parse("$(lowercase(problem))_test()"))
      end
    end
    @testset "Problem $(nlp.meta.name)" begin
      @info "$(problem) consistency"
      @testset "Consistency" begin
        @time consistent_nlps([nlp, nlp])
      end
      @info "$(problem) check dimension"
      @testset "Check dimensions" begin
        @time check_nlp_dimensions(nlp)
      end
      #@testset "Multiple precision support" begin
      #  multiple_precision_nlp(nlp)
      #end
      @info "$(problem) view"
      @testset "View subarray" begin
        @time view_subarray_nlp(nlp)
      end
      @info "$(problem) coord"
      if local_test # issue with windows and macos 1, because hess_coord is not ideal
        @testset "Test coord memory" begin
          @time coord_memory_nlp(nlp)
        end
      end
    end
  end
end

# Test constructors, util_functions.jl and additional_obj_terms.jl
include("unit-test.jl")
