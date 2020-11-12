module SQPFactFree

using NLPModels

include("SQP-NLPModel.jl")

export SQPNLP, obj, objgrad, objgrad!, grad!, grad, hess,
       cons, cons!, jprod, jprod!, jtprod, jtprod!,
       hprod, hprod!, hess_coord, hess_coord!,
       hess_structure, hess_structure!

using FastClosures, Krylov, LinearAlgebra, LinearOperators

 function solve_saddle_system(nlp :: AbstractNLPModel;
                              x0  :: AbstractVector{T} = nlp.meta.x0,
                              itmax :: Int = 100) where T

  #First step, we form the system
  # ( G  A' ) (x) = (-g)
  # ( A  0  ) (λ) = ( b)
  #G = hess(nlp, x0)
  #A = jac(nlp,  x0)
  #g = grad(nlp, 0.)
  #b = cons(nlp, 0.)

  rhs = vcat(- grad(nlp, zeros(nlp.meta.nvar)), cons(nlp, zeros(nlp.meta.nvar)))

  #We now create a LinearOperator
  jacop = jac_op(nlp, x0)
  Jv  = Array{T,1}(undef, nlp.meta.ncon)
  Jtv = Array{T,1}(undef, nlp.meta.nvar)
  prod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jacop'*v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon],
                            jacop*v[1:nlp.meta.nvar])
  ctprod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jacop'*v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon],
                            jacop*v[1:nlp.meta.nvar])
 #PreallocatedLinearOperator{T}(nrow::Int, ncol::Int, symmetric::Bool, hermitian::Bool, prod, tprod, ctprod)
  sad_op = PreallocatedLinearOperator{T}(nlp.meta.ncon+nlp.meta.nvar, nlp.meta.ncon+nlp.meta.nvar, true, true, prod, ctprod, ctprod)

  #(x, stats) = symmlq(sad_op, rhs)
  (x, stats) = minres(sad_op, rhs, itmax = itmax)

  return x, stats
 end

using SolverTools, Logging

export sqp_solver
"""
Bound constraints are ignored, and assume ucon=lcon=0
"""
 function sqp_solver(nlp  :: AbstractNLPModel;
                     x0   :: AbstractVector{T} = nlp.meta.x0,
                     atol :: AbstractFloat = 1e-3,
                     max_iter :: Int = 10,
                     itmax :: Int = 100,
                     lsfunc :: Function = armijo_wolfe) where T

  x = copy(x0)
  lqp = SQPNLP(nlp, copy(x0), zeros(nlp.meta.ncon))
  score = norm(lqp.gx, Inf)
  OK = score <= atol

  #h = LineModel(nlp, x, lqp.gx)

  @info log_header([:iter, :f, :c, :score, :sigma], [Int, T, T, T, T],
                   hdr_override=Dict(:f=>"f(x)", :c=>"||c(x)||", :score=>"‖∇L‖", :sigma=>"σ"))
  @info log_row(Any[0, lqp.fx, norm(lqp.cx,Inf), score, lqp.delta])

  i=0
  while !OK
   p, stats = solve_saddle_system(lqp, x0 = zeros(nlp.meta.nvar), itmax = itmax)
   if ~stats.solved @show stats.status end

   #redirect!(h, vcat(lqp.xk,lqp.lk), p)
   #slope = dot(nlp.meta.nvar, p, lqp.gx)
   # Perform improved Armijo linesearch.
   #t, good_grad, ft, nbk, nbW = armijo_wolfe(h, f, slope, ∇ft, τ₁=T(0.9999), bk_max=25, verbose=false)
   #t, good_grad, ft, nbk, nbW = lsfunc(h, lqp.fx, slope, lqp.gx, τ₁=T(0.9999), bk_max=25, verbose=false)

   lqp.xk += p[1:nlp.meta.nvar]
   lqp.lk += p[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon]

   i += 1
   score = norm(p[1:nlp.meta.nvar], Inf)
   OK = (score <= atol) || (i > max_iter)
   if !OK
    lqp.fx, lqp.gx, lqp.cx = obj(nlp, lqp.xk), grad(nlp, lqp.xk), cons(nlp, lqp.xk)
   end
   @info log_row(Any[i, lqp.fx, norm(lqp.cx,Inf), norm(score,Inf), lqp.delta])
  end #end of main loop

  #stats = GenericExecutionStats((score <= atol), nlp, solution = lqp.xk)

  return lqp.xk
 end

end #end of module
