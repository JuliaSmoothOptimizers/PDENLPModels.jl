# PDENLPModels Progress

The structures implemented in [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) are `AbstractNLPModel` and therefore
implement the traditional functions.

Note that (for now) there is no matrix-vector methods available in Gridap.
No performance check has been done yet (such as `ProfileView` or `@code_warntype`).

### Constructors
* [  ] The constructors handling functional bounds, and constraints on κ are missing. Also, I should rethink the constructors as nearly 50 are planned...

### Regarding the objective function

We handle four types of objective terms:
* [ ✓ ] `NoFETerm`
* [ ✓ ] `EnergyFETerm`
* [ ✓ ] `MixedEnergyFETerm` works for separated functional and discrete unknowns. The real mixed case is ongoing but significantly harder (now running - tests).
* [   ] `ResidualEnergyFETerm` **TODO**

The export connected implemented functions are
* [ ✓ ] `obj` checked.
* [ ✓ ] `grad!` Done with cellwise autodiff.
* [ ✓ ] `hess` For the objective function: computes the hessian cell by cell with autodiff, but construct only the lower triangular in sparse format.
* [ ✓ ] `hess_coord!` 
* [ ✓ ] `hess_obj_structure!` (not by default in `NLPModels`)
* [ ✓ ] `hprod!` For the objective function: call the hessian in coo format and then use `coo_sym_prod`.
* [ ✓ ] `hess_op!` see `hprod!`
* [ ✓ ] `hess_op` uses `hess_op!`

### Regarding the constraints

* [ ✓ ] `cons!` checked
* [ ✓ ] `jac` Done with cellwise autodiff, or analytical jacobian, or matrix in linear case. Sparse output.
* [ ✓ ] `jprod!`  Call `jac` and then compute the product.
* [ ✓ ] `jtprod!` Call `jac` and then compute the product.
* [ ✓ ] `jac_op`  Call `jac` and then compute the product.
* [ ✓ ] `jac_op!` Call `jac` and then compute the product.
* [ ✓ ] `jac_structure!` works well for AffineFEOperator and FEOperatorFromTermsr.
* [ ✓ ] `jac_coord!` (following `jac_structure!`)

### On the Lagrangian function

* [   ] `hess`  uses ForwardDiff on `obj(nlp, x)` - TODO: improve with SparseDiffTools - TODO: job done cell by cell, see [issue](https://github.com/gridap/Gridap.jl/issues/458) on Gridap.jl
* [   ] `hess_coord!` **TODO** (following `hess_structure!`)
* [   ] `hess_structure!` **TODO**
* [   ] `hprod!`  use ForwardDiff gradient-and-derivative on `obj(nlp, x)`.
* [   ] `hess_op!` see `hprod!`
* [ ✓ ] `hess_op` uses `hess_op!`
