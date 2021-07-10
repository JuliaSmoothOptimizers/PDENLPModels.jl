#This package
using Gridap, PDENLPModels
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
using NLPModels, NLPModelsIpopt, NLPModelsTest, Random, Test
using LinearAlgebra, SparseArrays

Random.seed!(1998)

const pde_problems = [
  "BURGER1D",
  "CELLINCREASE",
  "SIS",
  "CONTROLSIR",
  "DYNAMICSIR",
  "BASICUNCONSTRAINED",
  "PENALIZEDPOISSON",
  #"INCOMPRESSIBLENAVIERSTOKES", #too slow
  "POISSONMIXED",
  "POISSONPARAM",
  "POISSONMIXED2",
  "TOREBRACHISTOCHRONE",
  "CONTROLELASTICMEMBRANE",
]

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
      local_test || eval(Meta.parse("$(lowercase(problem))_test()"))
      x = rand(nlp.meta.nvar)
      obj(nlp, x)
      grad(nlp, x)
      hess_structure(nlp)
      hess_coord(nlp, x)
      hess(nlp, x)
      if nlp.meta.ncon > 0
        y = rand(nlp.meta.ncon)
        cons(nlp, x)
        jac(nlp, x)
        hess(nlp, x, y)
        hess_coord(nlp, x, y)
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
      @testset "Test coord memory" begin
        @time coord_memory_nlp(nlp)
      end
    end
  end
end

# Test constructors, util_functions.jl and additional_obj_terms.jl
include("unit-test.jl")
