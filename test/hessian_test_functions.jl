using Gridap, PDENLPModels, PDEOptimizationProblems, Test
using LinearAlgebra, SparseArrays, NLPModels

nlp = controlelasticmembrane1()

using PDENLPModels: count_hess_nnz_coo_short

"""
Function testing all the hessian-related function implemented for GridapPDENLPModel.

- return true if the test passed.
- set udc to true to use NLPModels derivative check (can be slow)
https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/dercheck.jl
- should be used for small problems only

We test the functions from hessian_func.jl and normal test functions.

List of functions we are testing:
- hess_coo ✓
- hess ✓
- hess_obj_structure ✓
- hess_coord (Implemented?)
- count_hess_nnz_coo_short ✓
- hprod
- hess_op and hess_op!

The Lagrangian Hessian will be tested later.

Comments:
* set the number of "nnzh" (with the repeated entries?) in the META? Note, this is
necessary to have hess_structure! and hess_coord!.
* `count_hess_nnz_coo` is used in the computation of hess_coo instead of
`count_hess_nnz_coo_short`, so if there is a difference it would appear in the
tests (implicit test).
* Use sparse differentiation
"""
function hessian_test_functions(nlp :: GridapPDENLPModel; udc = false)

  n = nlp.meta.nvar
  nnzh = nlp.meta.nnzh #Lagrangian function
  #Why not an nnzh for the Hessian of the objective function?

  a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  ncells = num_cells(nlp.tnrj.trian)
  cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

  nnz_hess_yu = count_hess_nnz_coo_short(a, cell_id_yu)
  #add the nnz w.r.t. k; by default it is:
  nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2) + (n - 1) * nlp.nparam
  @test nnzh >= nnz_hess_yu

  #Test the LowerTriangular:
  H1 = hess(nlp, nlp.meta.x0)
  xr = rand(nlp.meta.nvar)
  Hr = hess(nlp, xr)
  #the matrix is sparse
  @test issparse(H1)
  @test issparse(Hr)
  @test size(H1) == (n,n)
  @test size(Hr) == (n,n)

  (I1, J1, V1) = findnz(H1)
  (Ir, Jr, Vr) = findnz(Hr)
  #It's structure should be stable
  @test I1 == Ir && J1 == Jr

  (sI0, sJ0, sV0) = hess_coo(nlp, nlp.meta.x0, obj_weight = 0.0)
  @test !(false in (sV0 .== 0.0))

  (sI1, sJ1, sV1) = hess_coo(nlp, nlp.meta.x0, obj_weight = 0.5)
  (sIr, sJr, sVr) = hess_coo(nlp, xr, obj_weight = 0.9)
  (sI, sJ) = hess_obj_structure(nlp)
  @test sI1 == sIr && sI1 == sI0
  @test sJ1 == sJr && sJ1 == sJ0
  @test sI == sI1 && sJ == sJ1
  @test length(sI) == nnz_hess_yu
  @test length(sJ) == nnz_hess_yu
  #Note that this is in general different that coo format from hess as there
  #might be repeated entries.
  @test length(V1) <= length(sV1)
  @test length(Vr) <= length(sVr)

  H1bis = sparse(sI1, sJ1, sV1, nlp.meta.nvar, nlp.meta.nvar)
  Hrbis = sparse(sIr, sJr, sVr, nlp.meta.nvar, nlp.meta.nvar)

  @test H1bis == 0.5*H1
  @test Hrbis == 0.9*Hr

  @test_throws DimensionError hess(nlp, vcat(nlp.meta.x0, 1.))
  #hess_coo doesn't verify the size, but ignores x entries after nlp.meta.nvar.

  if nlp.nparam == 0

  else
    @warn "_hessian_test_functions: not finished for nparam > 0"
  end

  if udc
    @test gradient_check(nlp) == Dict{Int64,Float64}()
    H_errs = hessian_check(nlp) #slow
    @test H_errs[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
    H_errs_fg = hessian_check_from_grad(nlp)
    @test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
  end

  return true
end

hessian_test_functions(nlp)
