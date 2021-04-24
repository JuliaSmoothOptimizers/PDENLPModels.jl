using PDENLPModels, Gridap, NLPModelsTest, Test

include("problems/sis.jl")
include("problems/dynamic_sir.jl")
include("problems/control_sir.jl")
include("problems/cell_increase.jl")

test_problems = [Burger1d(n=10), sis(n=10), dynamic_sir(n=10), control_sir(n=10), cell_increase(n=10)]

@testset "NLP tests" begin
  for nlp in test_problems
    @testset "Problem $(nlp.meta.name)" begin
      @testset "Consistency" begin
        consistent_nlps([nlp, nlp])
      end
      @testset "Check dimensions" begin
        check_nlp_dimensions(nlp)
      end
      #@testset "Multiple precision support" begin
      #  multiple_precision_nlp(nlp)
      #end
      @testset "View subarray" begin
        view_subarray_nlp(nlp)
      end
      @testset "Test coord memory" begin
        coord_memory_nlp(nlp)
      end
    end
  end
end
