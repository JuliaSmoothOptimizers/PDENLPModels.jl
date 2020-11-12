import NLPModels: increment!, obj, objgrad, objgrad!, grad!, grad, hess, cons, cons!, jprod, jprod!, jtprod, jtprod!, hprod, hprod!, hess_coord, hess_coord!, hess_structure, hess_structure!
"""
We consider here the implementation of an sequential quadratic programming for
the minimization problem:
min\\_x f(x) s.t. c(x) = 0.

Given x^k This SQPNLP corresponds to

min\\_p fk + <∇fk, p> + 0.5<p,∇^2 Lk p> s.t. ∇ck p + ck = 0.

"""
mutable struct SQPNLP{S <: AbstractFloat, T <: AbstractVector{S}} <: AbstractNLPModel

    meta     :: AbstractNLPModelMeta
    counters :: Counters
    nlp      :: AbstractNLPModel

    xk   :: T
    lk   :: T

    #Evaluation of the SQPNLP functions contains info on nlp:
    fx  :: Union{S, T, Nothing}
    cx  :: Union{T, Nothing}
    gx  :: Union{T, Nothing}

    delta :: AbstractFloat #parameter used for cubic regularization

    function SQPNLP(meta, counters, nlp, xk, lk)
        S, T = eltype(nlp.meta.x0), typeof(nlp.meta.x0)
        fx = obj(nlp, xk)
        cx = cons(nlp, xk)
        gx = grad(nlp, xk)

        delta = 1e-2

        return new{S,T}(meta, counters, nlp, xk, lk, fx, cx, gx, delta)
    end
end

function SQPNLP(nlp :: AbstractNLPModel, xk, lk)
 return SQPNLP(nlp.meta, nlp.counters, nlp, xk, lk)
end

function grad!(nlp ::  SQPNLP, x :: AbstractVector{T}, gx :: AbstractVector{T}) where {T <: AbstractFloat}
 return nlp.gx + hprod(nlp.nlp, nlp.xk, nlp.lk, x)
end

function obj(nlp ::  SQPNLP, x :: AbstractVector{T}) where {T <: AbstractFloat}
 return nlp.fx + dot(x, nlp.gx) + 0.5 * dot(x, hprod(nlp.nlp, nlp.xk, nlp.lk, x))
end

function objgrad!(nlp :: SQPNLP, x :: AbstractVector{T}, gx :: AbstractVector{T}) where {T <: AbstractFloat}
 return obj(nlp, x), grad!(nlp, x, gx)
end

function cons!(nlp ::  SQPNLP, x :: AbstractVector{T}, c :: AbstractVector{T}) where {T <: AbstractFloat}
 c .= jprod(nlp.nlp, nlp.xk, x) + nlp.cx
 return  c
end

function jprod!(nlp :: SQPNLP, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)

  return jprod!(nlp.nlp, nlp.xk, v, Jv)
end

function jtprod!(nlp :: SQPNLP, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)

  return jtprod!(nlp.nlp, nlp.xk, v, Jtv)
end

"""
    hess_structure!(nlp, rows, cols)
Return the structure of the Lagrangian Hessian in sparse coordinate format in place.
"""
function hess_structure!(nlp :: SQPNLP, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  @lencheck nlp.meta.nnzh rows cols
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end


function hess_coord!(nlp :: SQPNLP, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)

  Hx = obj_weight * hess(nlp.nlp, nlp.xk, nlp.lk)

  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end

  return vals
end

function hess_coord!(nlp :: SQPNLP, x :: AbstractVector, y :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon y
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  #This is a linearly constrained optimization problem
  return hess_coord!(nlp, x, vals; obj_weight = obj_weight)
end

function hprod!(nlp :: SQPNLP, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector, Hv  :: AbstractVector; obj_weight=1.0)
 return hprod!(nlp, x, v, Hv, obj_weight = obj_weight)
end

function hprod!(nlp :: SQPNLP, x :: AbstractVector, v :: AbstractVector, Hv  :: AbstractVector; obj_weight=1.0)
 @lencheck nlp.meta.nvar x v Hv
 increment!(nlp, :neval_hprod)

 return hprod!(nlp.nlp, nlp.xk, nlp.lk, v, Hv)
end
