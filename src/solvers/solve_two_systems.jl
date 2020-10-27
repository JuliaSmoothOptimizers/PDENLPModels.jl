function _solve_system_dense(nlp  ::  FletcherPenaltyNLP, x :: AbstractVector{T}, rhs1, rhs2; kwargs...)  where T

  A =  NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
  I = diagm(0 => ones(nlp.meta.nvar)) #Matrix(I, nlp.meta.nvar, nlp.meta.nvar) or spdiagm(0 => ones(nlp.meta.nvar))
  M = [I A'; A 0] #expensive

  sol1 = M \ rhs1

  if rhs2 != nothing
      sol2 = M \ rhs2
  else
      sol2 = nothing
  end

  return sol1, sol2
end

function _solve_with_linear_operator(nlp  ::  FletcherPenaltyNLP, x :: AbstractVector{T}, rhs1, rhs2; _linear_system_solver :: Function = cg,  kwargs...)  where T

    #size(A) : nlp.nlp.meta.ncon x nlp.nlp.meta.nvar
    n = nlp.nlp.meta.ncon + nlp.nlp.meta.nvar
    #Tanj: Would it be beneficial to have a jjtprod returning Jv and Jtv ?
    Mp(v) = vcat(v[1:nlp.nlp.meta.nvar] + jtprod(nlp.nlp,x,v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.nlp.meta.ncon]),
                 jprod(nlp.nlp, x, v[1:nlp.nlp.meta.nvar]))
    #LinearOperator(type, nrows, ncols, symmetric, hermitian, prod, tprod, ctprod)
    opM = LinearOperator(Float64, n, n, true, true, v->Mp(v), w->Mp(w), u->Mp(u))

    (sol1, stats)  = _linear_system_solver(opM, rhs1; kwargs...)

    if rhs2 != nothing
        (sol2, stats)  = _linear_system_solver(opM, rhs2; kwargs...)
    else
        sol2 = nothing
    end

    return sol1, sol2
end

function _solve_system_factorization_eigenvalue(nlp, x :: AbstractVector{T}, rhs1, rhs2; kwargs...)  where T

        A =  NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
        I = diagm(0 => ones(nlp.meta.nvar))
        M = [I A'; A 0] #expensive
        O, Δ = eigen(M)#eigvecs(M), eigvals(M)
        # Boost negative values of Δ to 1e-8
        D = Δ .+ max.((1e-8 .- Δ), 0.0)

        sol1 = O*diagm(1.0 ./ D)*O'*rhs1

        if rhs2 != nothing
            sol2 = O*diagm(1.0 ./ D)*O'*rhs2
        else
            sol2 = nothing
        end

  return sol1, sol2
end

function _solve_system_factorization_lu(nlp, x :: AbstractVector{T}, rhs1, rhs2; kwargs...) where T

        n, ncon = nlp.meta.nvar, nlp.nlp.meta.ncon
        A = NLPModels.jac(nlp.nlp, x) #expensive (for large problems)
        In = Matrix{T}(I, n, n) #spdiagm(0 => ones(nlp.meta.nvar)) ?
        M = [In A'; A zeros(ncon, ncon)] #expensive
        LU = lu(M)

        sol1 = LU \ rhs1

        if rhs2 != nothing
            sol2 = LU \ rhs2
        else
            sol2 = nothing
        end

  return sol1, sol2
end
