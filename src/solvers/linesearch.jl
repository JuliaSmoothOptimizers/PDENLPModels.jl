"""
Same as https://github.com/JuliaSmoothOptimizers/SolverTools.jl/blob/master/src/linesearch/armijo_wolfe.jl
but only with objgrad
"""
function armijo_og(h :: LineModel,
                         h₀ :: T,
                         slope :: T,
                         g :: Array{T,1};
                         t :: T=one(T),
                         τ₀ :: T=max(T(1.0e-4), sqrt(eps(T))),
                         τ₁ :: T=T(0.9999),
                         bk_max :: Int=10,
                         bW_max :: Int=5,
                         verbose :: Bool=false) where T <: AbstractFloat

  # Perform improved Armijo linesearch.
  nbk = 0
  nbW = 0

  # First try to increase t to satisfy loose Wolfe condition
  ht, slope_t = objgrad!(h, t, g)
  while (slope_t < τ₁*slope) && (ht <= h₀ + τ₀ * t * slope) && (nbW < bW_max)
    t *= 5
    ht, slope_t = objgrad!(h, t, g)
    nbW += 1
  end

  hgoal = h₀ + slope * t * τ₀;
  fact = -T(0.8)
  ϵ = eps(T)^T(3/5)

  # Enrich Armijo's condition with Hager & Zhang numerical trick
  Armijo = (ht <= hgoal)
  good_grad = true
  while !Armijo && (nbk < bk_max)
    t *= T(0.4)
    ht, = obj(h, t)
    hgoal = h₀ + slope * t * τ₀;

    good_grad = false

    # avoids unused grad! calls
    Armijo = false
    if ht <= hgoal
      Armijo = true
    end

    nbk += 1
  end

  verbose && @printf("  %4d %4d\n", nbk, nbW);

  return (t, good_grad, ht, nbk, nbW)
end
