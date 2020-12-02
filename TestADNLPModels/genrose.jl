include("test_func.jl")

#Copy-paste from
#https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/test/problems/genrose.jl
"Generalized Rosenbrock model in size `n`"
function genrose_autodiff(n :: Int=500)

  n < 2 && error("genrose: number of variables must be ≥ 2")

  x0 = [i/(n+1) for i = 1:n]
  f(x::AbstractVector) = begin
    s = 1.0
    for i = 1:n-1
      s += 100 * (x[i+1]-x[i]^2)^2 + (x[i]-1)^2
    end
    return s
  end

  return ADNLPModel(f, x0, name="genrose_autodiff")
end

n=100
nlp = genrose_autodiff(n)

hv_test(nlp)

#=
#Compute the hessian:
sparsity_pattern = hessian_sparsity(f, nlp.meta.x0)
_H = Float64.(sparse(sparsity_pattern))
colors = matrix_colors(_H)

#Doesn't work for now...
#Hessian coloring support is coming soon! see Readme https://github.com/JuliaDiff/FiniteDiff.jl
#FiniteDiff.finite_difference_hessian!(_H, f, nlp.meta.x0, colorvec=colors)
=#
