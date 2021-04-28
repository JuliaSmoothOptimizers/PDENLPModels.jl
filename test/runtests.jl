using LinearAlgebra, SparseArrays
#This package
using Gridap
#PDENLPModels
using PDENLPModels
using PDENLPModels:
  FEFunctionType,
  _split_vector,
  _split_FEFunction,
  _obj_integral,
  _obj_cell_integral,
  _compute_gradient_k,
  _compute_gradient!,
  _compute_hess_coo
#Testing
using NLPModels, NLPModelsTest, Test

const pde_problems = [
  "BURGER1D",
  # "CELLINCREASE",
  # "SIS",
  # "CONTROLSIR",
  # "DYNAMICSIR",
  "BASICUNCONSTRAINED",
  "PENALIZEDPOISSON",
  "INCOMPRESSIBLENAVIERSTOKES",
  "POISSONMIXED",
] #use upper-case

for problem in pde_problems
  include("problems/$(lowercase(problem)).jl")
end

n = 3
#Tanj: for each problem there is a lowercase(problem) function that returns a GridapPDENLPModel
#++ would be to also have a lowercase(problem)_test that test the problem with the exact solution.
local_test = false

@testset "NLP tests" begin
  for problem in pde_problems
    @info "$(problem)"
    @time nlp = eval(Meta.parse("$(lowercase(problem))(n=$(n))"))
    @testset "Test problem scenario" begin
      # local_test || eval(Meta.parse("$(lowercase(problem))_test()"))
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
      @testset "Test coord memory" begin
      @time coord_memory_nlp(nlp)
      end
    end
  end
end

# Test constructors, util_functions.jl and additional_obj_terms.jl
include("unit-test.jl")
