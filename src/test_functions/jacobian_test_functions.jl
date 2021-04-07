"""
Function testing all the jacobian-related function implemented for GridapPDENLPModel.

- return true if the test passed.
- set `udc` to true to use NLPModels derivative check (can be slow)
https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/dercheck.jl
- should be used for small problems only as it computes hessians several times.

We test the functions from jacobian_func.jl and normal test functions.

List of functions we are testing:
- `jac` ✓
- `jac_structure` and  `jac_structure!` ✓
- `jac_coord` and `jac_coord!` ✓
- `count_nnz_jac` ✓
- `jprod` and `jtprod` 
- `jac_op` and `jac_op!` ✓

Comments:
* set the number of "nnzh" (with the repeated entries?) in the META?
* Use sparse differentiation
"""
function jacobian_test_functions(nlp :: GridapPDENLPModel; udc = false, tol = 1e-15)
    
  n, ncon = nlp.meta.nvar, nlp.meta.ncon
    
  nnzj_yu = Main.PDENLPModels.count_nnz_jac(nlp.op, nlp.Y, nlp.Xpde, nlp.Ypde, nlp.Ycon)
  nnz_jac_k = nlp.nparam > 0 ? ncon * nlp.nparam : 0
  nnzj = nnzj_yu + nnz_jac_k
    
  @test nnzj == nlp.meta.nnzj
   
  Jac1 = jac(nlp, nlp.meta.x0)
  xr = rand(n)
  Jacr = jac(nlp, xr)
    
  @test issparse(Jac1)
  @test issparse(Jacr)
  @test size(Jac1) == (ncon, n)
  @test size(Jacr) == (ncon, n)
    
  (I1, J1) = jac_structure(nlp)
  V1  = jac_coord(nlp, nlp.meta.x0)
  Vr  = jac_coord(nlp, xr)
  @test length(I1) == nnzj
  @test length(J1) == nnzj
  @test length(V1) == nnzj
  @test length(Vr) == nnzj
    
    
  _J1 = sparse(I1, J1, V1, nlp.meta.ncon, nlp.meta.nvar)
  _Jr = sparse(I1, J1, Vr, nlp.meta.ncon, nlp.meta.nvar)
    
  @test Jac1 ≈ _J1 atol = tol
  @test Jacr ≈ _Jr atol = tol
    
  @test_throws DimensionError jac(nlp, vcat(nlp.meta.x0, 1.))
  @test_throws DimensionError jac_coord(nlp, vcat(nlp.meta.x0, 1.))
    
  _Jxz  = jprod(nlp, nlp.meta.x0, zeros(nlp.meta.nvar))
  _Jxv  = jprod(nlp, nlp.meta.x0, ones(nlp.meta.nvar))
  _Jxvr = jprod(nlp, xr, ones(nlp.meta.nvar))
  @test _Jxz == zeros(nlp.meta.ncon)
  @test _Jxv ≈ Jac1 * ones(nlp.meta.nvar) atol=tol
  @test _Jxvr ≈ Jacr * ones(nlp.meta.nvar)      atol=tol
    
  @test_throws DimensionError jprod(nlp, vcat(nlp.meta.x0, 1.), nlp.meta.x0)
  @test_throws DimensionError jprod(nlp, nlp.meta.x0, vcat(nlp.meta.x0, 1.))
    
  _Jtxz  = jtprod(nlp, nlp.meta.x0, zeros(nlp.meta.ncon))
  _Jtxv  = jtprod(nlp, nlp.meta.x0, ones(nlp.meta.ncon))
  _Jtxvr = jtprod(nlp, xr, ones(nlp.meta.ncon))
  @test _Jtxz == zeros(nlp.meta.nvar)
  @test _Jtxv ≈ Jac1' * ones(nlp.meta.ncon) atol=tol
  @test _Jtxvr ≈ Jacr' * ones(nlp.meta.ncon)      atol=tol
    
  @test_throws DimensionError jtprod(nlp, vcat(nlp.meta.x0, 1.), nlp.meta.y0)
  @test_throws DimensionError jtprod(nlp, nlp.meta.x0, vcat(nlp.meta.y0, 1.))
    
  jop1 = jac_op(nlp, nlp.meta.x0)
  jopr = jac_op(nlp, xr)
  @test _Jxz == jop1 * zeros(nlp.meta.nvar)
  @test _Jxv ≈ jop1 * ones(nlp.meta.nvar)  atol=tol
  @test _Jxvr ≈ jopr * ones(nlp.meta.nvar) atol=tol
    
  if udc
    @test jacobian_check(nlp) == Dict{Tuple{Int64,Int64},Float64}()
  end
  return true
end
