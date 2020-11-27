# PDENLPModels Progress

The structures implemented in [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) are `AbstractNLPModel` and therefore
implement the traditional functions.

Note that (for now) there is no matrix-vector methods available in Gridap.
No performance check has been done yet (such as `ProfileView` or `@code_warntype`).

### Regarding the objective function

We handle four types of objective terms:
* [x] `NoFETerm`
* [x] `EnergyFETerm`
* [ ] `MixedEnergyFETerm` fix the issue.
* [ ] `ResidualEnergyFETerm` **TODO**

The export connected implemented functions are
* [x] `obj` checked.
* [x] `grad!` Done with cellwise autodiff.
* [x] `hess` For the objective function: computes the hessian cell by cell with autodiff, but construct only the lower triangular in sparse format.
* [ ] `hess_coord!` **TODO** (following `hess_structure!`)
* [ ] `hess_structure!` **TODO**
* [x] `hprod!` for the objective function and the Lagrangian. For the objective function: call the hessian in coo format and then use `coo_sym_prod`.
* [x] `hess_op!` see `hprod!`
* [x] `hess_op` uses `hess_op!`

### Regarding the constraints

* [x] `cons!` checked
* [x] `jac` Done with cellwise autodiff, or analytical jacobian, or matrix in linear case. Sparse output.
* [x] `jprod!`  Call `jac` and then compute the product.
* [x] `jtprod!` Call `jac` and then compute the product.
* [x] `jac_op`  Call `jac` and then compute the product.
* [x] `jac_op!` Call `jac` and then compute the product.
* [ ] `jac_structure!` works well for AffineFEOperator; **TODO** for nonlinear.
* [ ] `jac_coord!` **TODO** (following `jac_structure!`)

### On the Lagrangian function

* [ ] `hess`  uses ForwardDiff on `obj(nlp, x)` - TODO: improve with SparseDiffTools - TODO: job done cell by cell, see [issue](https://github.com/gridap/Gridap.jl/issues/458) on Gridap.jl
* [ ] `hess_coord!` **TODO** (following `hess_structure!`)
* [ ] `hess_structure!` **TODO**
* [ ] `hprod!`  use ForwardDiff gradient-and-derivative on `obj(nlp, x)`.
* [ ] `hess_op!` see `hprod!`
* [x] `hess_op` uses `hess_op!`
