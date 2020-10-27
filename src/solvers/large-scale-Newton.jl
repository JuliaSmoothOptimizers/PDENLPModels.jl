#copy paste from https://abelsiqueira.github.io/blog/nlpmodelsjl-and-cutestjl-constrained-optimization/
#M. Lalee, J. Nocedal, and T. Plantenga. On the implementation of an algorithm for large-scale equality constrained optimization. SIAM J. Optim., Vol. 8, No. 3, pp. 682-706, 1998.
#Let’s implement a “simple” solver for constrained optimization. Our solver will loosely follow the Byrd-Omojokun implementation of

function solver(nlp :: AbstractNLPModel)
  if !equality_constrained(nlp)
    error("This solver is for equality constrained problems")
  elseif has_bounds(nlp)
    error("Can't handle bounds")
  end

  x = nlp.meta.x0

  fx = obj(nlp, x)
  cx = cons(nlp, x)

  ∇fx = grad(nlp, x)
  Jx = jac_op(nlp, x)

  λ = cgls(Jx', -∇fx)[1]
  ∇ℓx = ∇fx + Jx'*λ
  norm∇ℓx = norm(∇ℓx)

  Δ = max(0.1, min(100.0, 10norm∇ℓx))
  μ = 1
  v = zeros(nlp.meta.nvar)

  iter = 0
  while (norm∇ℓx > 1e-4 || norm(cx) > 1e-4) && (iter < 10000)
    # Vertical step
    if norm(cx) > 1e-4
      v = cg(Jx'*Jx, -Jx'*cx, radius=0.8Δ)[1]
      Δp = sqrt(Δ^2 - dot(v,v))
    else
      fill!(v, 0)
      Δp = Δ
    end

    # Horizontal step
    # Simplified to consider only ∇ℓx = proj(∇f, Nu(A))
    B = hess_op(nlp, x, y=λ)
    B∇ℓx = B * ∇ℓx
    gtBg = dot(∇ℓx, B∇ℓx)
    gtγ = dot(∇ℓx, ∇fx + B * v)
    t = if gtBg <= 0
      norm∇ℓx > 0 ? Δp/norm∇ℓx : 0.0
    else
      t = min(gtγ/gtBg, Δp/norm∇ℓx)
    end

    d = v - t * ∇ℓx

    # Trial step acceptance
    xt = x + d
    ft = obj(nlp, xt)
    ct = cons(nlp, xt)
    γ = dot(d, ∇fx) + 0.5*dot(d, B * d)
    θ = norm(cx) - norm(Jx * d + cx)
    normλ = norm(λ, Inf)
    if θ <= 0
      μ = normλ
    elseif normλ > γ/θ
      μ = min(normλ, 0.1 + γ/θ)
    else
      μ = 0.1 + γ/θ
    end
    Pred = -γ + μ * θ
    Ared = fx - ft + μ * (norm(cx) - norm(ct))

    ρ = Ared/Pred
    if ρ > 1e-2
      x .= xt
      fx = ft
      cx .= ct
      ∇fx = grad(nlp, x)
      Jx = jac_op(nlp, x)
      λ = cgls(Jx', -∇fx)[1]
      ∇ℓx = ∇fx + Jx'*λ
      norm∇ℓx = norm(∇ℓx)
      if ρ > 0.75 && norm(d) > 0.99Δ
        Δ *= 2.0
      end
    else
      Δ *= 0.5
    end

    iter += 1
  end

  return x, fx, norm∇ℓx, norm(cx)
end
