"""
Function testing all the Lagrangian Hessian-related function implemented for GridapPDENLPModel.

- return true if the test passed.
- set `udc` to true to use NLPModels derivative check (can be slow)
https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/dercheck.jl
- should be used for small problems only as it computes hessians several times.

We test the functions from hessian_func.jl and normal test functions.

List of functions we are testing:
- `hess_coo` ✓
- `hess` ✓
- `hess_obj_structure` ✓
- `hess_coord` and `hess_coord!` ✓
- `hprod` ✓
- `hess_op` and `hess_op!` ✓
"""
function hessian_lagrangian_test_functions(nlp::GridapPDENLPModel; udc = false, tol = 1e-15)
  n, m, p = nlp.meta.nvar, nlp.meta.ncon, nlp.nparam
  nnzh = nlp.meta.nnzh #Lagrangian function
  #Why not an nnzh for the Hessian of the objective function?
  x0, y0 = nlp.meta.x0, nlp.meta.y0
  xr, yr = rand(n), rand(m)

  Hx = hess(nlp, x0)
  Hr = hess(nlp, xr)

  v0 = hess_coord(nlp, x0)
  vr = hess_coord(nlp, xr)
  (rows, cols) = hess_obj_structure(nlp)
  nnzh_obj = length(rows)
  #=
  #Si 1 term, et même trian:
  if length(nlp.op.terms) == 1
    @test nnzh == 2 * nnzh_obj 
  end
  =#

  #Compare when Lagrangian vanishes:
  Hx0 = hess(nlp, x0, zeros(m))
  Hxr = hess(nlp, xr, zeros(m))
  Hxr2 = hess(nlp, xr, zeros(m), obj_weight = 0.5)

  @test Hx0 == Hx
  @test Hxr == Hr
  @test Hxr2 == hess(nlp, xr, obj_weight = 0.5)

  #The old dense function (very slow)
  function hess2(
    nlp::GridapPDENLPModel,
    x::AbstractVector,
    λ::AbstractVector;
    obj_weight::Real = one(eltype(x)),
  )
    @lencheck nlp.meta.nvar x
    @lencheck nlp.meta.ncon λ
    function ℓ(x, λ)
      κ, xyu = x[1:(nlp.nparam)], x[(nlp.nparam + 1):(nlp.meta.nvar)]
      yu = FEFunction(nlp.Y, xyu)
      int = PDENLPModels._obj_integral(nlp.tnrj, κ, yu)

      c = similar(x, nlp.meta.ncon)
      PDENLPModels._from_terms_to_residual!(nlp.op, x, nlp, c)

      return obj_weight * sum(int) + dot(c, λ)
    end

    #ℓ(x) = obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)
    Hx = ForwardDiff.hessian(x -> ℓ(x, λ), x)
    return tril(Hx)
  end

  Hxy = hess(nlp, x0, y0)
  Hxyr = hess(nlp, xr, yr)
  Hxyr2 = hess(nlp, xr, yr, obj_weight = 0.5)
  if udc
    @test Hxy == hess2(nlp, x0, y0)
    @test Hxyr == hess2(nlp, xr, yr)
    @test Hxyr2 == hess2(nlp, xr, yr, obj_weight = 0.5)
  end

  @test issparse(Hxy)
  @test issparse(Hxyr)
  @test size(Hxy) == (n, n)
  @test size(Hxyr) == (n, n)
  (I0, J0, V0) = findnz(Hxy)
  (Ir, Jr, Vr) = findnz(Hxyr)
  @test I0 == Ir && J0 == Jr

  (rows, cols) = hess_structure(nlp)
  @test (length(rows) == nnzh) && length(cols) == nnzh
  vxy0 = hess_coord(nlp, x0, y0)
  vxyr = hess_coord(nlp, xr, yr)
  @test length(vxy0) == nnzh
  @test length(vxyr) == nnzh
  sH0 = sparse(rows, cols, vxy0, n, n)
  sHr = sparse(rows, cols, vxyr, n, n)
  @test sH0 == Hxy
  @test sHr == Hxyr

  (sI0, sJ0, sV0) = hess_coo(nlp, nlp.op, xr, yr) #this is just constraints
  nnzh_obj
  (sI, sJ) = hess_structure(nlp)
  sV = hess_coord(nlp, xr, yr)
  @test sI[(nnzh_obj + 1):end] == sI0
  @test sJ[(nnzh_obj + 1):end] == sJ0
  @test sV[(nnzh_obj + 1):end] == sV0

  _Hxz = hprod(nlp, x0, y0, zeros(n))
  _Hxv = hprod(nlp, x0, y0, ones(n), obj_weight = 0.5)
  _Hxvr = hprod(nlp, xr, yr, ones(n))
  Hxyr = hess(nlp, xr, yr)
  @test _Hxz == zeros(nlp.meta.nvar)
  @test _Hxv ≈ 0.5 * Symmetric(Hxy, :L) * ones(n)
  @test _Hxvr ≈ Symmetric(Hxyr, :L) * ones(n)

  hop1 = hess_op(nlp, x0, y0, obj_weight = 0.5)
  hopr = hess_op(nlp, xr, yr)
  @test _Hxz == hop1 * zeros(n)
  @test _Hxv ≈ hop1 * ones(n)
  @test _Hxvr ≈ hopr * ones(n)

  return true
end
