import JSOSolvers.lbfgs
function lbfgs(stp :: NLPStopping; x0 :: AbstractVector{T} = stp.current_state.x, atol = 1e-3, mem = 5, lsfunc = SolverTools.armijo_wolfe) where T

 (x, f, ∇f, H) = lbfgs(stp.pb, x0, atol = atol, mem = mem, lsfunc = lsfunc)
 Stopping.update!(stp.current_state, x = x, fx = f, gx = ∇f, Hx = H)

 return stp
end

"""
Adaptation from lbfgs of JSOSolvers.jl
https://github.com/JuliaSmoothOptimizers/JSOSolvers.jl/blob/master/src/lbfgs.jl

- handle objgrad whenever possible
- Stopping buffer
- adapt output
- remove stats and info-log (maybe put it back)
"""
function lbfgs(nlp ::  FletcherPenaltyNLP, x0 :: AbstractVector{T}; atol = 1e-3, mem = 5, lsfunc = armijo_og) where T

     n = nlp.meta.nvar

     xt = zeros(T, n)
     ∇ft = zeros(T, n)

 x = copy(x0)
 f, ∇f = objgrad(nlp, x)
 H = InverseLBFGSOperator(T, n, mem, scaling=true)
 h = LineModel(nlp, x, ∇f) #SolverTools.jl

 iter = 0
 OK = norm(∇f) <= atol


  while !OK
    d = - H * ∇f
    slope = dot(n, d, ∇f)
    if slope ≥ 0
      @error "not a descent direction" slope
      status = :not_desc
      stalled = true
      continue
    end

    redirect!(h, x, d)
    # Perform improved Armijo linesearch.
    #t, good_grad, ft, nbk, nbW = armijo_wolfe(h, f, slope, ∇ft, τ₁=T(0.9999), bk_max=25, verbose=false)
    t, good_grad, ft, nbk, nbW = lsfunc(h, f, slope, ∇ft, τ₁=T(0.9999), bk_max=25, verbose=false)

    copyaxpy!(n, t, d, x, xt)
    good_grad || grad!(nlp, xt, ∇ft)

    # Update L-BFGS approximation.
    push!(H, t * d, ∇ft - ∇f)

    # Move on.
    x .= xt
    f = ft
    ∇f .= ∇ft

    ∇fNorm = nrm2(n, ∇f)
    iter = iter + 1

    OK = (∇fNorm <= atol) || iter >= 100
  end
 return x, f, ∇f, H
end
