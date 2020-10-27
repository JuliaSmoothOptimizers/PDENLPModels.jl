using Test

include("../../src/solvers/Fletcher-equality-constrained-penalty.jl")
include("../../src/solvers/large-scale-Newton.jl")


using CUTEst

function runcutest()
  problems = filter(x->length(x) <= 5, CUTEst.select(only_free_var=true, only_equ_con=true))
  sort!(problems)
  @printf("%-7s  %15s  %15s  %15s\n",
          "Problem", "f(x)", "‖∇ℓ(x,λ)‖", "‖c(x)‖")
  for p in problems
    nlp = CUTEstModel(p)
    try
      x, fx, nlx, ncx = solver(nlp)
      @printf("%-7s  %15.8e  %15.8e  %15.8e\n", p, fx, nlx, ncx)
    catch
      @printf("%-7s  %s\n", p, "failure")
    finally
      finalize(nlp)
    end
  end
end

clp = CUTEstModel("BT1")
cons(clp, clp.meta.x0)
jac(clp, clp.meta.x0)
hess(clp, clp.meta.x0, y=clp.meta.y0)
finalize(clp)
