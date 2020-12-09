var documenterSearchIndex = {"docs":
[{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"","category":"page"},{"location":"KrylovforLinearPDE/#JSOPDESolver","page":"Krylov.jl to solve linear PDE","title":"JSOPDESolver","text":"","category":"section"},{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"using Gridap, Krylov","category":"page"},{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"Set of codes to use JSOSolvers tools to solve partial differential equations modeled with Gridap. It contains examples of:","category":"page"},{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"using Krylov.jl to solve linear PDEs (AffineFEOperator)\nusing Krylov.jl to solve the linear systems in the Newton-loop to solve","category":"page"},{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"nonlinear PDEs.","category":"page"},{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"more...","category":"page"},{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"Pluto notebook examples can be found in the pluto folder.","category":"page"},{"location":"KrylovforLinearPDE/#Krylov.jl-to-solve-linear-PDEs","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDEs","text":"","category":"section"},{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"include(\"header.jl\")\n\n###############################################################################\n#Gridap resolution:\n#This corresponds to a Poisson equation with Dirichlet and Neumann conditions\n#described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/\nfunction _poisson()\n    domain = (0,1,0,1)\n    n = 2^7\n    partition = (n,n)\n    model = CartesianDiscreteModel(domain,partition)\n\n    trian = Triangulation(model)\n    degree = 2\n    quad = CellQuadrature(trian,degree)\n\n    V0 = TestFESpace(\n      reffe=:Lagrangian, order=1, valuetype=Float64,\n      conformity=:H1, model=model, dirichlet_tags=\"boundary\")\n\n    g(x) = 0.0\n    Ug = TrialFESpace(V0,g)\n\n    w(x) = 1.0\n    a(u,v) = ∇(v)⊙∇(u)\n    b_Ω(v) = v*w\n    t_Ω = AffineFETerm(a,b_Ω,trian,quad)\n\n    op_pde = AffineFEOperator(Ug,V0,t_Ω)\n    return op_pde\nend\n\nop_pde = _poisson()\n\n#Gridap.jl/src/FESpaces/FESolvers.jl\n#Gridap.jl/src/Algebra/LinearSolvers.jl\n@time ls  = KrylovSolver(minres; itmax = 150)\n@time ls1 = LUSolver()\n@time ls2 = BackslashSolver()\n\nsolver  = LinearFESolver(ls)\nsolver1 = LinearFESolver(ls1)\nsolver2 = LinearFESolver(ls2)\n\n#Describe the matrix:\n@test size(get_matrix(op_pde)) == (16129, 16129)\n@test issparse(get_matrix(op_pde))\n@test issymmetric(get_matrix(op_pde))\n\nuh  = solve(solver, op_pde)\nuh1 = solve(solver1,op_pde)\nuh2 = solve(solver2,op_pde)\n#Sad, that we don't have the stats back...\n\n@time uh = solve(solver,op_pde)\nx = get_free_values(uh)\n@time uh1 = solve(solver1,op_pde)\nx1 = get_free_values(uh1)\n@time uh2 = solve(solver2,op_pde)\nx2 = get_free_values(uh2)\n\n@test norm(x  - x1, Inf) <= 1e-8\n@test norm(x1 - x2, Inf) <= 1e-13\n@show norm(get_matrix(op_pde)*x  - get_vector(op_pde),Inf) <= 1e-8\n@test norm(get_matrix(op_pde)*x1 - get_vector(op_pde),Inf) <= 1e-15\n@test norm(get_matrix(op_pde)*x2 - get_vector(op_pde),Inf) <= 1e-15","category":"page"},{"location":"KrylovforLinearPDE/#Krylov.jl-to-solve-the-linear-systems-in-the-Newton-loop-to-solve-nonlinear-PDEs","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve the linear systems in the Newton-loop to solve nonlinear PDEs","text":"","category":"section"},{"location":"KrylovforLinearPDE/","page":"Krylov.jl to solve linear PDE","title":"Krylov.jl to solve linear PDE","text":"include(\"header.jl\")\n\nop = _pdeonlyincompressibleNS()\n\nusing LineSearches\n#Gridap way of solving the equation:\nnls = NLSolver(\n  show_trace=false, method=:newton, linesearch=BackTracking())\nsolver = FESolver(nls)\n\n#struct NewtonRaphsonSolver <:NonlinearSolver\n#  ls::LinearSolver\n#  tol::Float64\n#  max_nliters::Int\n#end\nnls2 = Gridap.Algebra.NewtonRaphsonSolver(LUSolver(), 1e-6, 100)\nsolver2 = FESolver(nls2)\n\n#The first approach is to use Newton method anticipated by Gridap and using\n#Krylov.jl to solve the linear problem.\n#NLSolver(ls::LinearSolver;kwargs...)\nls  = KrylovSolver(cgls; itmax = 10000, verbose = false)\nnls_krylov = NLSolver(ls, show_trace=false)\n@test nls_krylov.ls == ls\nsolver_krylov = FESolver(nls_krylov)\n\nnls_krylov2 = Gridap.Algebra.NewtonRaphsonSolver(ls, 1e-6, 100)\nsolver_krylov2 = FESolver(nls_krylov2)\n\n#Another version is to surcharge:\n#solve!(x::AbstractVector,nls::NewNonlinearSolverType,op::NonlinearOperator,cache::Nothing)\n\n#\n# Finally, we solve the problem:\n#solve(solver, op)\n#solve(solver2, op)\n#solve(solver_krylov, op)\n\n@time uph1 = solve(solver,op)\nsol_gridap1 = get_free_values(uph1);\n@time uph2 = solve(solver2,op)\nsol_gridap2 = get_free_values(uph2);\n@time uph3 = solve(solver_krylov,op)\nsol_gridap3 = get_free_values(uph3);\n@time uph4 = solve(solver_krylov2,op)\nsol_gridap4 = get_free_values(uph4);\n\nnUg = num_free_dofs(op.trial)\n@test size(Gridap.FESpaces.jacobian(op, uph1)) == (nUg, nUg)\n\n@show norm(Gridap.FESpaces.residual(op, uph1),Inf)\n@show norm(Gridap.FESpaces.residual(op, uph2),Inf)\n@show norm(Gridap.FESpaces.residual(op, uph3),Inf)\n@show norm(Gridap.FESpaces.residual(op, uph4),Inf)\n\n@show norm(sol_gridap1 - sol_gridap2, Inf)\n@show norm(sol_gridap1 - sol_gridap3, Inf)\n@show norm(sol_gridap1 - sol_gridap4, Inf)","category":"page"},{"location":"nlpmodels/#PDENLPModels-Progress","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"","category":"section"},{"location":"nlpmodels/","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"The structures implemented in NLPModels are AbstractNLPModel and therefore implement the traditional functions.","category":"page"},{"location":"nlpmodels/","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"Note that (for now) there is no matrix-vector methods available in Gridap. No performance check has been done yet (such as ProfileView or @code_warntype).","category":"page"},{"location":"nlpmodels/#Regarding-the-objective-function","page":"PDENLPModels Progress","title":"Regarding the objective function","text":"","category":"section"},{"location":"nlpmodels/","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"We handle four types of objective terms:","category":"page"},{"location":"nlpmodels/","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"[x] NoFETerm\n[x] EnergyFETerm\n[ ] MixedEnergyFETerm fix the issue.\n[ ] ResidualEnergyFETerm TODO","category":"page"},{"location":"nlpmodels/","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"The export connected implemented functions are","category":"page"},{"location":"nlpmodels/","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"[x] obj checked.\n[x] grad! Done with cellwise autodiff.\n[x] hess For the objective function: computes the hessian cell by cell with autodiff, but construct only the lower triangular in sparse format.\n[ ] hess_coord! TODO (following hess_structure!)\n[ ] hess_structure! TODO\n[x] hprod! For the objective function: call the hessian in coo format and then use coo_sym_prod.\n[x] hess_op! see hprod!\n[x] hess_op uses hess_op!","category":"page"},{"location":"nlpmodels/#Regarding-the-constraints","page":"PDENLPModels Progress","title":"Regarding the constraints","text":"","category":"section"},{"location":"nlpmodels/","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"[x] cons! checked\n[x] jac Done with cellwise autodiff, or analytical jacobian, or matrix in linear case. Sparse output.\n[x] jprod!  Call jac and then compute the product.\n[x] jtprod! Call jac and then compute the product.\n[x] jac_op  Call jac and then compute the product.\n[x] jac_op! Call jac and then compute the product.\n[ ] jac_structure! works well for AffineFEOperator; TODO for nonlinear.\n[ ] jac_coord! TODO (following jac_structure!)","category":"page"},{"location":"nlpmodels/#On-the-Lagrangian-function","page":"PDENLPModels Progress","title":"On the Lagrangian function","text":"","category":"section"},{"location":"nlpmodels/","page":"PDENLPModels Progress","title":"PDENLPModels Progress","text":"[ ] hess  uses ForwardDiff on obj(nlp, x) - TODO: improve with SparseDiffTools - TODO: job done cell by cell, see issue on Gridap.jl\n[ ] hess_coord! TODO (following hess_structure!)\n[ ] hess_structure! TODO\n[ ] hprod!  use ForwardDiff gradient-and-derivative on obj(nlp, x).\n[ ] hess_op! see hprod!\n[x] hess_op uses hess_op!","category":"page"},{"location":"#PDENLPModels.jl-Documentation","page":"Introduction","title":"PDENLPModels.jl Documentation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"","category":"page"},{"location":"#Functions","page":"Introduction","title":"Functions","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Modules = [PDENLPModels]","category":"page"},{"location":"#PDENLPModels.EnergyFETerm","page":"Introduction","title":"PDENLPModels.EnergyFETerm","text":"FETerm modeling the objective function of the optimization problem.\n\nbeginequation\nint_Omega f(yu) dOmega\nendequation\n\nwhere Ω is described by:\n\ntrian :: Triangulation\nquad  :: CellQuadrature\n\nConstructor:\n\nEnergyFETerm(:: Function, :: Triangulation, :: CellQuadrature)\n\nSee also: MixedEnergyFETerm, NoFETerm, _obj_cell_integral, _obj_integral, compute\\gradient_k!\n\n\n\n\n\n","category":"type"},{"location":"#PDENLPModels.GridapPDENLPModel","page":"Introduction","title":"PDENLPModels.GridapPDENLPModel","text":"PDENLPModels using Gridap.jl\n\nhttps://github.com/gridap/Gridap.jl Cite: Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia. Journal of Open Source Software, 5(52), 2520.\n\nFind functions (y,u): Y -> ℜⁿ x ℜⁿ and κ ∈ ℜⁿ satisfying\n\nmin      ∫_Ω​ f(κ,y,u) dΩ​ s.t.     y solution of a PDE(κ,u)=0          lcon <= c(κ,y,u) <= ucon          lvar <= (κ,y,u)  <= uvar\n\n     ```math\n     \\begin{aligned}\n     \\min_{κ,y,u} \\ & ∫_Ω​ f(κ,y,u) dΩ​ \\\\\n     \\mbox{ s.t. } & y \\mbox{ solution of } PDE(κ,u)=0, \\\\\n     & lcon <= c(κ,y,u) <= ucon, \\\\\n     & lvar <= (κ,y,u)  <= uvar.\n     \\end{aligned}\n     ```\n\nThe weak formulation is then: res((y,u),(v,q)) = ∫ v PDE(κ,y,u) + ∫ q c(κ,y,u)\n\nwhere the unknown (y,u) is a MultiField see Tutorials 7  and 8 of Gridap.\n\nThe set Ω​ is represented here with trian and quad.\n\nTODO: [ ] time evolution pde problems.    [ ] Handle the case where g and H are given.    [ ] Handle several terms in the objective function (via an FEOperator)?    [ ] Be more explicit on the different types of FETerm in  fromtermtoterms!    [ ] Could we control the Dirichlet boundary condition? (like classical control of heat equations)    [ ] Clean the tests.    [ ] Missing: constraint ncon with numfreedofs(Xpde)?   \n\nMain constructor:\n\nGridapPDENLPModel(:: NLPModelMeta, :: Counters, :: AbstractEnergyTerm, :: FESpace, :: Union{FESpace,Nothing}, :: FESpace, :: Union{FESpace,Nothing}, :: FESpace, :: Union{FESpace,Nothing}, :: Union{FEOperator, Nothing}, :: Int, :: Int, :: Int)\n\nAdditional constructors:\n\nUnconstrained and no control\n\nGridapPDENLPModel(x0, tnrj, Ypde, Xpde)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde)     GridapPDENLPModel(f, trian, quad, Ypde, Xpde)   \n\nBound constraints:   \n\nGridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar)     GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lvar, uvar)   \n\nPDE-constrained:   \n\nGridapPDENLPModel(x0, tnrj, Ypde, Xpde, c)     GridapPDENLPModel(Ypde, Xpde, c)     GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c)     GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, c)     GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon)     GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon)   \n\nPDE-constrained and bounds:   \n\nGridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, c)  GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c)     GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c)     GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon)     GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon)   \n\nFuture constructors:\n\nFunctional bounds: in this case |lvar|=|uvar|=nparam\n\nGridapPDENLPModel(x0, tnrj, Ypde, Xpde, lfunc, ufunc)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lfunc, ufunc)     GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lfunc, ufunc)  GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, lfunc, ufunc)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar, lfunc, ufunc)     GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lvar, uvar, lfunc, ufunc)   GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, lfunc, ufunc, c)  GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c)     GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c)     GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon)     GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon)     \n\nDiscrete constraints (ck, lckon, uckon) only for problems with nparam > 0 (hence only if x0 given or tnrj)\n\nGridapPDENLPModel(x0, tnrj, Ypde, Xpde, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, ck, lckon, uckon)   GridapPDENLPModel(x0, tnrj, Ypde, Xpde, c, ck, lckon, uckon)   GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, ck, lckon, uckon)  GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon, ck, lckon, uckon)    GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, c, ck, lckon, uckon)  GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, ck, lckon, uckon)   GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon, ck, lckon, uckon)   GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lfunc, ufunc, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lfunc, ufunc, ck, lckon, uckon)   GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, lfunc, ufunc, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar, lfunc, ufunc, ck, lckon, uckon)  GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, lfunc, ufunc, c, ck, lckon, uckon)  GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, ck, lckon, uckon)  GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon, ck, lckon, uckon)     GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon, ck, lckon, uckon)\n\nThe following keyword arguments are available to all constructors:\n\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\nlin: An array of indexes of the linear constraints\n\n(default: Int[] or 1:ncon if c is an AffineFEOperator)\n\nThe following keyword arguments are available to the constructors for constrained problems explictly giving lcon and ucon:\n\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\nNotes:\n\nWe handle two types of FEOperator: AffineFEOperator, and FEOperatorFromTerms\n\nwhich is the obtained by the FEOperator constructor.  The terms supported in FEOperatorFromTerms are: FESource, NonlinearFETerm,  NonlinearFETermWithAutodiff, LinearFETerm, AffineFETerm.\n\nIf lcon and ucon are not given, they are assumed zeros.\nIf the type can't be deduced from the argument, it is Float64.\n\n\n\n\n\n","category":"type"},{"location":"#PDENLPModels.MixedEnergyFETerm","page":"Introduction","title":"PDENLPModels.MixedEnergyFETerm","text":"FETerm modeling the objective function of the optimization problem with functional and discrete unknowns.\n\nbeginequation\nint_Omega f(yukappa) dOmega\nendequation\n\nwhere Ω is described by:\n\ntrian :: Triangulation\nquad  :: CellQuadrature\n\nConstructor:\n\nMixedEnergyFETerm(:: Function, :: Triangulation, :: CellQuadrature, :: Int)\n\nSee also: EnergyFETerm, NoFETerm, _obj_cell_integral, _obj_integral, _compute_gradient_k!\n\n\n\n\n\n","category":"type"},{"location":"#PDENLPModels.NoFETerm","page":"Introduction","title":"PDENLPModels.NoFETerm","text":"FETerm modeling the objective function when there are no intregral objective.\n\nmath \\begin{equation}  f(\\kappa) \\end{equation}\n\nConstructors:\n\nNoFETerm()\n\nNoFETerm(:: Function)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, _obj_cell_integral, _obj_integral, _compute_gradient_k!\n\n\n\n\n\n","category":"type"},{"location":"#PDENLPModels.ResidualEnergyFETerm","page":"Introduction","title":"PDENLPModels.ResidualEnergyFETerm","text":"FETerm modeling the objective function of the optimization problem with functional and discrete unknowns, describe as a norm and a regularizer.\n\nbeginequation\nfrac12Fyu(yu)^2_L^2_Omega + lambdaint_Omega lyu(yu) dOmega\n + frac12Fk(κ)^2 + mu lk(κ)\nendequation\n\nwhere Ω is described by:\n\ntrian :: Triangulation\nquad  :: CellQuadrature\n\nConstructor:\n\nResidualEnergyFETerm(:: Function, :: Triangulation, :: CellQuadrature, :: Function, :: Int)\n\nSee also: EnergyFETerm, NoFETerm, MixedEnergyFETerm\n\n\n\n\n\n","category":"type"},{"location":"#NLPModels.jprod!-Tuple{GridapPDENLPModel,AbstractArray{T,1} where T,AbstractArray{T,1} where T,AbstractArray{T,1} where T}","page":"Introduction","title":"NLPModels.jprod!","text":"Jv = jprod!(nlp, x, v, Jv)\n\nEvaluate J(x)v, the Jacobian-vector product at x in place.\n\nNote for GridapPDENLPModel:\n\nEvaluate the jacobian and then use mul! (here coo_prod! is slower as we have to compute findnz).\nAlternative: benefit from the AD? Jv .= ForwardDiff.derivative(t->nlp.c(nlp, x + t * v), 0)\n\nwhen the jacobian is obtained by AD.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jtprod!-Tuple{GridapPDENLPModel,AbstractArray{T,1} where T,AbstractArray{T,1} where T,AbstractArray{T,1} where T}","page":"Introduction","title":"NLPModels.jtprod!","text":"Jv = jtprod!(nlp, x, v, Jv)\n\nEvaluate J(x)v, the Jacobian-vector product at x in place.\n\nNote for GridapPDENLPModel:\n\nEvaluate the jacobian and then use mul! (here coo_prod! is slower as we have to compute findnz).\nAlternative: benefit from the AD? Jtv .= ForwardDiff.gradient(x -> dot(nlp.c(x), v), x)\n\nwhen the jacobian is obtained by AD.\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._compute_gradient!","page":"Introduction","title":"PDENLPModels._compute_gradient!","text":"Return the gradient of the objective function and set it in place.\n\n_compute_gradient!(:: AbstractVector, :: EnergyFETerm, :: AbstractVector, :: FEFunctionType, :: FESpace, :: FESpace)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _obj_cell_integral, _compute_hess_coo, _compute_hess_k_coo\n\n\n\n\n\n","category":"function"},{"location":"#PDENLPModels._compute_gradient_k","page":"Introduction","title":"PDENLPModels._compute_gradient_k","text":"Return the derivative of the objective function w.r.t. κ.\n\n_compute_gradient_k(:: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _obj_cell_integral, _compute_hess_coo, _compute_hess_k_coo\n\n\n\n\n\n","category":"function"},{"location":"#PDENLPModels._compute_hess_coo","page":"Introduction","title":"PDENLPModels._compute_hess_coo","text":"Return the hessian w.r.t. yu of the objective function in coo format.\n\n_compute_hess_coo(:: AbstractEnergyTerm, :: AbstractVector, :: FEFunctionType, :: FESpace, :: FESpace)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _obj_cell_integral, _compute_gradient_k, _compute_hess_k_coo\n\n\n\n\n\n","category":"function"},{"location":"#PDENLPModels._compute_hess_k_coo","page":"Introduction","title":"PDENLPModels._compute_hess_k_coo","text":"Return the hessian w.r.t. κ of the objective function in coo format.\n\n_compute_hess_k_coo(:: AbstractNLPModel, :: AbstractEnergyTerm, :: AbstractVector, :: AbstractVector)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _obj_cell_integral, _compute_gradient_k, _compute_hess_coo\n\n\n\n\n\n","category":"function"},{"location":"#PDENLPModels._compute_hess_k_vals","page":"Introduction","title":"PDENLPModels._compute_hess_k_vals","text":"Return the values of the hessian w.r.t. κ of the objective function.\n\n_compute_hess_k_vals(:: AbstractNLPModel, :: AbstractEnergyTerm, :: AbstractVector, :: AbstractVector)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _obj_cell_integral, _compute_gradient_k, _compute_hess_coo\n\n\n\n\n\n","category":"function"},{"location":"#PDENLPModels._convert_bound_functions_to_bounds-Tuple{AbstractArray{T,1} where T,Gridap.FESpaces.FESpace,Nothing,Function,Function}","page":"Introduction","title":"PDENLPModels._convert_bound_functions_to_bounds","text":"TO BE FINISHED:\n\nconvert bounds function as bounds vectors?\nbecareful of multi-field functions\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._fill_hess_at_cell!-Union{Tuple{Vi}, Tuple{Ii}, Tuple{M}, Tuple{Type{M},Any,Array{Ii,1},Array{Ii,1},Array{Vi,1},Any,Any,Any,Any}} where Vi<:AbstractFloat where Ii<:Int64 where M","page":"Introduction","title":"PDENLPModels._fill_hess_at_cell!","text":"https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/SparseMatrixAssemblers.jl#L463 fillmatrixatcell! may have a specific specialization\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._from_terms_to_jacobian-Union{Tuple{T}, Tuple{Gridap.FESpaces.FEOperatorFromTerms,AbstractArray{T,1},Gridap.FESpaces.FESpace,Gridap.FESpaces.FESpace,Gridap.FESpaces.FESpace,Union{Nothing, Gridap.FESpaces.FESpace}}} where T<:Number","page":"Introduction","title":"PDENLPModels._from_terms_to_jacobian","text":"Note:\n\nCompute the derivatives w.r.t. y and u separately.\nUse AD for those derivatives. Only for the following:\n\nNonlinearFETerm (we neglect the inapropriate jac function);\nNonlinearFETermWithAutodiff\nTODO: Gridap.FESpaces.FETerm & AffineFETerm ?\nFESource <: AffineFETerm (jacobian of a FESource is nothing)\nLinearFETerm <: AffineFETerm (not implemented)\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._from_terms_to_jacobian2-Union{Tuple{T}, Tuple{Gridap.FESpaces.FEOperatorFromTerms,AbstractArray{T,1},GridapPDENLPModel}} where T<:AbstractFloat","page":"Introduction","title":"PDENLPModels._from_terms_to_jacobian2","text":"Would be better but somehow autodiffcelljacobianfromresidual is restricted to square matrices at some point.\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._from_terms_to_residual!-Tuple{Gridap.FESpaces.AffineFEOperator,AbstractArray{T,1} where T,GridapPDENLPModel,AbstractArray{T,1} where T}","page":"Introduction","title":"PDENLPModels._from_terms_to_residual!","text":"Note:\n\nmul! seems faster than doing:\n\nrows, cols, vals = findnz(getmatrix(op)) cooprod!(cols, rows, vals, v, res)\n\nget_matrix(op) is a sparse matrix\nBenchmark equivalent to Gridap.FESpaces.residual!(res, op_affine.op, xrand)\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._obj_cell_integral","page":"Introduction","title":"PDENLPModels._obj_cell_integral","text":"Return the integral of the objective function\n\n_obj_cell_integral(:: AbstractEnergyTerm, :: GenericCellField,  :: AbstractVector)\n\nx is a vector of GenericCellField, for instance resulting from yuh = CellField(Y, cell_yu).\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _compute_gradient_k, _compute_hess_coo, _compute_hess_k_coo\n\n\n\n\n\n","category":"function"},{"location":"#PDENLPModels._obj_integral","page":"Introduction","title":"PDENLPModels._obj_integral","text":"Return the integral of the objective function\n\n_obj_integral(:: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_cell_integral, _compute_gradient_k, _compute_hess_coo, _compute_hess_k_coo\n\n\n\n\n\n","category":"function"},{"location":"#PDENLPModels._split_FEFunction-Tuple{AbstractArray{T,1} where T,Gridap.FESpaces.FESpace,Gridap.FESpaces.FESpace}","page":"Introduction","title":"PDENLPModels._split_FEFunction","text":"_split_FEFunction(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})\n\nSplit the vector x into two FEFunction corresponding to the solution y and the control u. Returns nothing for the control u if Ycon == nothing.\n\nDo not verify the compatible length.\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._split_vector-Tuple{AbstractArray{T,1} where T,Gridap.FESpaces.FESpace,Gridap.FESpaces.FESpace}","page":"Introduction","title":"PDENLPModels._split_vector","text":"_split_vector(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})\n\nSplit the vector x into three vectors: y, u, k. Returns nothing for the control u if Ycon == nothing.\n\nDo not verify the compatible length.\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._struct_hess_at_cell!-Union{Tuple{Ii}, Tuple{M}, Tuple{Type{M},Any,Array{Ii,1},Array{Ii,1},Any,Any,Any,Any,Any}} where Ii<:Int64 where M","page":"Introduction","title":"PDENLPModels._struct_hess_at_cell!","text":"https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/SparseMatrixAssemblers.jl#L463 fillmatrixatcell! may have a specific specialization\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels._vals_hess_at_cell!-Union{Tuple{Vi}, Tuple{M}, Tuple{Type{M},Int64,Array{Vi,1},Any,Any,Any,Any}} where Vi<:AbstractFloat where M","page":"Introduction","title":"PDENLPModels._vals_hess_at_cell!","text":"https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/SparseMatrixAssemblers.jl#L463 fillmatrixatcell! may have a specific specialization\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels.assemble_hess-Union{Tuple{T}, Tuple{Gridap.FESpaces.GenericSparseMatrixAssembler,T,Gridap.Arrays.IdentityVector{Int64}}} where T<:AbstractArray","page":"Introduction","title":"PDENLPModels.assemble_hess","text":"These functions: https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/SparseMatrixAssemblers.jl#L463\n\nhttps://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/Algebra/SparseMatrixCSC.jl\n\nhttps://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/Algebra/SparseMatrices.jl#L29-L33\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels.hess_obj_structure-Tuple{GridapPDENLPModel}","page":"Introduction","title":"PDENLPModels.hess_obj_structure","text":"hess_structure returns the sparsity pattern of the Lagrangian Hessian  in sparse coordinate format, and hess_obj_structure is only for the objective function hessian.\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels.hess_old-Tuple{GridapPDENLPModel,AbstractArray{T,1} where T}","page":"Introduction","title":"PDENLPModels.hess_old","text":"julia> @btime hess(nlp, sol_gridap);   614.938 ms (724536 allocations: 90.46 MiB)\n\njulia> @btime hessold(nlp, solgridap);   643.689 ms (724599 allocations: 127.48 MiB)\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels.hessian_test_functions-Tuple{GridapPDENLPModel}","page":"Introduction","title":"PDENLPModels.hessian_test_functions","text":"Function testing all the hessian-related function implemented for GridapPDENLPModel.\n\nreturn true if the test passed.\nset udc to true to use NLPModels derivative check (can be slow)\n\nhttps://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/dercheck.jl\n\nshould be used for small problems only as it computes hessians several times.\n\nWe test the functions from hessian_func.jl and normal test functions.\n\nList of functions we are testing:\n\nhess_coo ✓\nhess ✓\nhess_obj_structure and  hess_obj_structure! ✓\nhess_coord and hess_coord! ✓\ncount_hess_nnz_coo_short ✓\nhprod ✓\nhess_op and hess_op! ✓\n\nComments:\n\nset the number of \"nnzh\" (with the repeated entries?) in the META? Note, this is\n\nnecessary to have hess_structure! and hess_coord!.\n\ncount_hess_nnz_coo is used in the computation of hess_coo instead of\n\ncount_hess_nnz_coo_short, so if there is a difference it would appear in the tests (implicit test).\n\nUse sparse differentiation\n\n\n\n\n\n","category":"method"},{"location":"#PDENLPModels.hprod_autodiff!-Tuple{GridapPDENLPModel,AbstractArray{T,1} where T,AbstractArray{T,1} where T,AbstractArray{T,1} where T}","page":"Introduction","title":"PDENLPModels.hprod_autodiff!","text":"Compute hessian-vector product of the objective function.\n\nNote: this is not efficient at all. Test on n=14115 @btime hprod(nlp, solgridap, v)   42.683 s (274375613 allocations: 29.89 GiB) while computing the hessian and then the product yields @btime _Hx = hess(nlp, solgridap);   766.036 ms (724829 allocations: 121.84 MiB) @btime hprod(nlp, sol_gridap, v)   42.683 s (274375613 allocations: 29.89 GiB)\n\n\n\n\n\n","category":"method"},{"location":"#Index","page":"Introduction","title":"Index","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"","category":"page"},{"location":"#References","page":"Introduction","title":"References","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"https://github.com/gridap/Gridap.jl Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia. Journal of Open Source Software, 5(52), 2520.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"https://github.com/JuliaSmoothOptimizers/NLPModels.jl D. Orban and A. S. Siqueira and {contributors} (2020). NLPModels.jl: Data Structures for Optimization Models","category":"page"}]
}
