using LinearAlgebra, SparseArrays
#This package
using Gridap
#PDENLPModels
using PDENLPModels
#=GRIDAPv15
using PDENLPModels:
  FEFunctionType,
  _split_vector,
  _split_FEFunction,
  _obj_integral,
  _obj_cell_integral,
  _compute_gradient_k,
  _compute_gradient!,
  _compute_hess_coo
=#
#Testing
using NLPModels, NLPModelsTest, Test

#=GRIDAPv15
const pde_problems = [
  "BURGER1D",
  # "CELLINCREASE",
  # "SIS",
  # "CONTROLSIR",
  # "DYNAMICSIR",
  "BASICUNCONSTRAINED",
  "PENALIZEDPOISSON",
  #"INCOMPRESSIBLENAVIERSTOKES", #too slow
  "POISSONMIXED",
  "POISSONPARAM",
  "POISSONMIXED2",
  "TOREBRACHISTOCHRONE",
  "CONTROLELASTICMEMBRANE",
]
=#
const pde_problems = [
  "BURGER1D", # OK
  # "CELLINCREASE", # TODO
  # "SIS", # TODO
  # "CONTROLSIR", # TODO
  # "DYNAMICSIR", # TODO
  "BASICUNCONSTRAINED", # OK
  "PENALIZEDPOISSON", # OK
  "INCOMPRESSIBLENAVIERSTOKES", #too slow # OK (except lagrangian-hess)
  # "POISSONMIXED", # TODO
  # "POISSONPARAM", # TODO
  #"POISSONMIXED2", # TODO
  "TOREBRACHISTOCHRONE", # OK
  "CONTROLELASTICMEMBRANE", # OK
]
# missing an example with an FESource term
# FEOperatorsFromTerms including a LinearFETerm

#=
Issue when computing derivatives with parameter.

typeof(xyu) = Vector{ForwardDiff.Dual{ForwardDiff.Tag{PDENLPModels.var"#89#91"{PDENLPModels.var"#_cons#90", GridapPDENLPModel{MixedEnergyFETerm}, Vector{Float64}}, Float64}, Float64, 2}}
Test problem scenario: Error During Test at /home/tmigot/.julia/dev/PDENLPModels.jl/test/runtests.jl:68
  Got exception outside of a @test
  AssertionError: 
  
  The entries stored in free_values and dirichlet_values should be of the same type.
  
  This error shows up e.g. when trying to build a FEFunction from a vector of integers
  if the Dirichlet values of the underlying space are of type Float64, or when the
  given free values are Float64 and the Dirichlet values ComplexF64.
  
  Stacktrace:
    [1] macro expansion
      @ ~/.julia/packages/Gridap/EZQEK/src/Helpers/Macros.jl:60 [inlined]
    [2] scatter_free_and_dirichlet_values(f::Gridap.FESpaces.UnconstrainedFESpace{Vector{Float64}}, free_values::Vector{ForwardDiff.Dual{ForwardDiff.Tag{PDENLPModels.var"#89#91"{PDENLPModels.var"#_cons#90", GridapPDENLPModel{MixedEnergyFETerm}, Vector{Float64}}, Float64}, Float64, 2}}, dirichlet_values::Vector{Float64})
      @ Gridap.FESpaces ~/.julia/packages/Gridap/EZQEK/src/FESpaces/UnconstrainedFESpaces.jl:38
    [3] scatter_free_and_dirichlet_values
      @ ~/.julia/packages/Gridap/EZQEK/src/FESpaces/TrialFESpaces.jl:108 [inlined]
    [4] FEFunction(fs::TrialFESpace{Gridap.FESpaces.UnconstrainedFESpace{Vector{Float64}}}, free_values::Vector{ForwardDiff.Dual{ForwardDiff.Tag{PDENLPModels.var"#89#91"{PDENLPModels.var"#_cons#90", GridapPDENLPModel{MixedEnergyFETerm}, Vector{Float64}}, Float64}, Float64, 2}}, dirichlet_values::Vector{Float64})
      @ Gridap.FESpaces ~/.julia/packages/Gridap/EZQEK/src/FESpaces/SingleFieldFESpaces.jl:160
    [5] FEFunction(fe::TrialFESpace{Gridap.FESpaces.UnconstrainedFESpace{Vector{Float64}}}, free_values::Vector{ForwardDiff.Dual{ForwardDiff.Tag{PDENLPModels.var"#89#91"{PDENLPModels.var"#_cons#90", GridapPDENLPModel{MixedEnergyFETerm}, Vector{Float64}}, Float64}, Float64, 2}})
      @ Gridap.FESpaces ~/.julia/packages/Gridap/EZQEK/src/FESpaces/SingleFieldFESpaces.jl:167
    [6] _from_terms_to_residual!(op::Gridap.FESpaces.FEOperatorFromWeakForm, x::Vector{ForwardDiff.Dual{ForwardDiff.Tag{PDENLPModels.var"#89#91"{PDENLPModels.var"#_cons#90", GridapPDENLPModel{MixedEnergyFETerm}, Vector{Float64}}, Float64}, Float64, 2}}, nlp::GridapPDENLPModel{MixedEnergyFETerm}, res::Vector{ForwardDiff.Dual{ForwardDiff.Tag{PDENLPModels.var"#89#91"{PDENLPModels.var"#_cons#90", GridapPDENLPModel{MixedEnergyFETerm}, Vector{Float64}}, Float64}, Float64, 2}})
      @ PDENLPModels ~/.julia/dev/PDENLPModels.jl/src/GridapPDENLPModel.jl:348
=#

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

#=GRIDAPv15
# Test constructors, util_functions.jl and additional_obj_terms.jl
include("unit-test.jl")
=#