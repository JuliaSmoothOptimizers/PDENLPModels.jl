var documenterSearchIndex = {"docs":
[{"location":"poisson-boltzman/#PDE-contrained-optimization","page":"PDE-constrained optimization","title":"PDE-contrained optimization","text":"","category":"section"},{"location":"poisson-boltzman/#Poisson-Boltzman-problem","page":"PDE-constrained optimization","title":"Poisson-Boltzman problem","text":"","category":"section"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"In this tutorial, we solve a control problem where the constraint is a 2D Poisson-Boltzman equation:","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"leftlbrace\nbeginaligned\nminlimits_y in H_0^1(Omega) u in L^2(Omega)     frac12int_Omega y - y_d(x)^2 + frac12alpha int_Omega u^2dx \ntext st   -Delta y + sinhy = h + u quad textin  Omega=(-11)^2\n                   y = 0 quad textin  partialOmega\nendaligned\nright","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"with the forcing term h(x_1x_2)=-sin(omega x_1) sin(omega x_2), omega = pi - frac18, and target state","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"beginaligned\n    y_d(x) = begincases\n    10 quad textif  x in 025075^2 \n    5 quad textotherwise\n    endcases\nendaligned","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"We refer to [1] for more details on Poisson-Boltzman equations.","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"The implementation as a GridapPDENLPModel is given as follows.","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"    using Gridap, PDENLPModels\n    #Domain\n    n = 100\n    model = CartesianDiscreteModel((-1,1,-1,1), (n,n))\n\n    #Definition of the spaces:\n    order = 1\n    valuetype = Float64\n    reffe = ReferenceFE(lagrangian, valuetype, order)\n    Xpde = TestFESpace(\n      model,\n      reffe;\n      conformity = :H1,\n      dirichlet_tags=\"boundary\",\n    )\n    Ypde = TrialFESpace(Xpde, 0.0)\n    Xcon = TestFESpace(model, reffe; conformity = :L2)\n    Ycon = TrialFESpace(Xcon)\n    Y = MultiFieldFESpace([Ypde, Ycon])\n\n    #Integration machinery\n    trian = Triangulation(model)\n    degree = 1\n    dΩ = Measure(trian,degree)\n\n    #Objective function:\n    yd(x) = min(x[1]-0.25, 0.75-x[1],x[2]-0.25, 0.75-x[2])>=0. ? 10. : 5.\n    function f(y, u)\n        ∫( 0.5 * (yd - y) * (yd - y) + 0.5 * 1e-4 * u * u )dΩ\n    end\n\n    #Definition of the constraint operator\n    ω = π - 1/8\n    h(x) = - sin(ω*x[1])*sin(ω*x[2])\n    function res(y, u, v)\n     ∫( ∇(v) ⊙ ∇(y) + (sinh ∘ y)*v - u*v - v * h )dΩ\n    end\n    op = FEOperator(res, Y, Xpde)\n    xin = zeros(Gridap.FESpaces.num_free_dofs(Y))\n    nlp = GridapPDENLPModel(xin, f, trian, Ypde, Ycon, Xpde, Xcon, op)","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"Then, one can solve the problem with Ipopt via NLPModelsIpopt.jl and plot the solution as a VTK file.","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"using NLPModelsIpopt\nstats = ipopt(nlp, print_level = 0)","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"Switching again the discrete solution as a FEFunction the result can written as a VTK-file using Gridap's facilities.","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"yfv = stats.solution[1:Gridap.FESpaces.num_free_dofs(nlp.pdemeta.Ypde)]\nyh  = FEFunction(nlp.pdemeta.Ypde, yfv)\nufv = stats.solution[1+Gridap.FESpaces.num_free_dofs(nlp.pdemeta.Ypde):end]\nuh  = FEFunction(nlp.pdemeta.Ycon, ufv)\nwritevtk(nlp.pdemeta.tnrj.trian,\"results\",cellfields=[\"uh\"=>uh, \"yh\"=>yh])","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"Finally, the solution is obtained using any software reading VTK, e.g. Paraview.","category":"page"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"(Image: Solution of P-B equation)(Image: Control of P-B equation)","category":"page"},{"location":"poisson-boltzman/#References","page":"PDE-constrained optimization","title":"References","text":"","category":"section"},{"location":"poisson-boltzman/","page":"PDE-constrained optimization","title":"PDE-constrained optimization","text":"[1] Michael J. Holst The Poisson-Boltzmann equation: Analysis and multilevel numerical solution. Applied Mathematics and CRPC, California Institute of Technology. (1994). Hols94d.pdf","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Contents","page":"Reference","title":"Contents","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Pages = [\"reference.md\"]","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"​","category":"page"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [PDENLPModels]","category":"page"},{"location":"reference/#PDENLPModels.EnergyFETerm","page":"Reference","title":"PDENLPModels.EnergyFETerm","text":"AbstractEnergyTerm modeling the objective function of the optimization problem\n\nbeginaligned\nint_Omega f(yu) dOmega\nendaligned\n\nConstructor:\n\nEnergyFETerm(:: Function, :: Triangulation, :: Measure)\n\nSee also: MixedEnergyFETerm, NoFETerm, _obj_cell_integral, _obj_integral, _compute_gradient_k!\n\n\n\n\n\n","category":"type"},{"location":"reference/#PDENLPModels.GridapPDENLPModel","page":"Reference","title":"PDENLPModels.GridapPDENLPModel","text":"GridapPDENLPModel returns an instance of an AbstractNLPModel using Gridap.jl for the discretization of the domain with finite-elements. Given a domain Ω ⊂ ℜᵈ Find a state function y: Ω -> Y, a control function u: Ω -> U and an algebraic vector κ ∈ ℜⁿ satisfying\n\nbeginaligned\nmin_κyu   _Ω f(κyu) dΩ \nmbox st   y mbox solution of  PDE(κu)=0 \n lcon = c(κyu) = ucon \n lvar = (κyu)  = uvar\nendaligned\n\nThe weak formulation of the PDE is then: res((y,u),(v,q)) = ∫ v PDE(κ,y,u) + ∫ q c(κ,y,u)\n\nwhere the unknown (y,u) is a MultiField see Tutorials 7  and 8 of Gridap.\n\nConstructors\n\nMain constructor:\n\nGridapPDENLPModel(::NLPModelMeta, ::Counters, ::PDENLPMeta, ::PDEWorkspace)\n\nThis is the main constructors with the attributes of the GridapPDENLPModel:\n\nmeta::NLPModelMeta: usual meta for NLPModels, see doc here;\ncounters::Counters: usual counters for NLPModels, see doc here;\npdemeta::PDENLPMeta: metadata specific to GridapPDENLPModel;\nworkspace::PDEWorkspace: Pre-allocated memory for GridapPDENLPModel.\n\nMore practical constructors are also available.\n\nFor unconstrained problems:\nGridapPDENLPModel(x0, ::Function, ::Union{Measure, Triangulation}, Ypde::FESpace, Xpde::FESpace; kwargs...)\nGridapPDENLPModel(x0, ::AbstractEnergyTerm, ::Union{Measure, Triangulation}, Ypde::FESpace, Xpde::FESpace; kwargs...)\nFor constrained problems without controls:\nGridapPDENLPModel(x0, ::Function, ::Union{Measure, Triangulation}, Ypde::FESpace, Xpde::FESpace, c::Union{Function, FEOperator}; kwargs...)\nGridapPDENLPModel(x0, ::AbstractEnergyTerm, Ypde::FESpace, Xpde::FESpace, c::Union{Function, FEOperator}; kwargs...)\nFor general constrained problems:\nGridapPDENLPModel(x0, ::Function, ::Union{Measure, Triangulation}, Ypde::FESpace, Ycon::FESpace, Xpde::FESpace, Xcon::FESpace, c::Union{Function, FEOperator}; kwargs...)\nGridapPDENLPModel(x0, ::AbstractEnergyTerm, Ypde::FESpace, Ycon::FESpace, Xpde::FESpace, Xcon::FESpace, c::Union{Function, FEOperator}; kwargs...)\n\nwhere the different arguments are:\n\nx0: initial guess for the system of size ≥ num_free_dofs(Ypde) + num_free_dofs(Ycon);\nf: objective function, the number of arguments depend on the application (y) or (y,u) or (y,u,θ);\nYpde: trial space for the state;\nYcon: trial space for the control (VoidFESpace if none);\nXpde: test space for the state;\nXcon: test space for the control (VoidFESpace if none);\nc: operator/function for the PDE-constraint, were we assume by default that the right-hand side is zero (otw. use lcon and ucon keywords), the number of arguments depend on the application (y,v) or (y,u,v) or (y,u,θ,v).\n\nIf length(x0) > num_free_dofs(Ypde) + num_free_dofs(Ycon), then the additional components are considered algebraic variables.\n\nThe function f and c must return integrals complying with Gridap's functions with a Measure/Triangulation given in the arguments of GridapPDENLPModel. Internally, the objective function f and the Measure/Triangulation are combined to instantiate an AbstractEnergyTerm.\n\nThe following keyword arguments are available to all constructors:\n\nname: The name of the model (default: \"Generic\")\n\nThe following keyword arguments are available to the constructors for constrained problems:\n\nlin: An array of indexes of the linear constraints (default: Int[])\n\nThe following keyword arguments are available to the constructors for constrained problems explictly giving lcon and ucon:\n\ny0: An inital estimate to the Lagrangian multipliers (default: zeros)\n\nThe bounds on the variables are given as AbstractVector via keywords arguments as well:\n\neither with lvar and uvar, or,\nlvary, lvaru, lvark, and uvary, uvaru, uvark.\n\nNotes:\n\nWe handle two types of FEOperator: AffineFEOperator, and FEOperatorFromWeakForm.\nIf lcon and ucon are not given, they are assumed zeros.\nIf the type can't be deduced from the argument, it is Float64.\n\nExample\n\nusing Gridap, PDENLPModels\n\n  # Definition of the domain\n  n = 100\n  domain = (-1, 1, -1, 1)\n  partition = (n, n)\n  model = CartesianDiscreteModel(domain, partition)\n\n  # Definition of the spaces:\n  valuetype = Float64\n  reffe = ReferenceFE(lagrangian, valuetype, 2)\n  Xpde = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = \"boundary\")\n  y0(x) = 0.0\n  Ypde = TrialFESpace(Xpde, y0)\n\n  reffe_con = ReferenceFE(lagrangian, valuetype, 1)\n  Xcon = TestFESpace(model, reffe_con; conformity = :H1)\n  Ycon = TrialFESpace(Xcon)\n\n  # Integration machinery\n  trian = Triangulation(model)\n  degree = 1\n  dΩ = Measure(trian, degree)\n\n  # Objective function:\n  yd(x) = -x[1]^2\n  α = 1e-2\n  function f(y, u)\n    ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u) * dΩ\n  end\n\n  # Definition of the constraint operator\n  ω = π - 1 / 8\n  h(x) = -sin(ω * x[1]) * sin(ω * x[2])\n  function res(y, u, v)\n    ∫(∇(v) ⊙ ∇(y) - v * u - v * h) * dΩ\n  end\n\n  # initial guess\n  npde = num_free_dofs(Ypde)\n  ncon = num_free_dofs(Ycon)\n  xin = zeros(npde + ncon)\n\n  nlp = GridapPDENLPModel(xin, f, trian, Ypde, Ycon, Xpde, Xcon, res, name = \"Control elastic membrane\")\n\nYou can also check the tutorial Solve a PDE-constrained optimization problem on JSO's website, juliasmoothoptimizers.github.io.\n\nWe refer to the folder test/problems for more examples of problems of different types:\n\ncalculus of variations,\noptimal control problem,\nPDE-constrained problems,\nmixed PDE-contrained problems with both function and algebraic unknowns. \n\nAn alternative is to visit the repository PDEOptimizationProblems that contains a collection of test problems.\n\nWithout objective function, the problem reduces to a classical PDE and we refer to Gridap tutorials for examples.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PDENLPModels.MixedEnergyFETerm","page":"Reference","title":"PDENLPModels.MixedEnergyFETerm","text":"AbstractEnergyTerm modeling the objective function of the optimization problem with functional and discrete unknowns\n\nbeginaligned\nint_Omega f(yukappa) dOmega\nendaligned\n\nConstructor:\n\nMixedEnergyFETerm(:: Function, :: Triangulation, :: Int)\n\nSee also: EnergyFETerm, NoFETerm, _obj_cell_integral, _obj_integral, _compute_gradient_k!\n\n\n\n\n\n","category":"type"},{"location":"reference/#PDENLPModels.NoFETerm","page":"Reference","title":"PDENLPModels.NoFETerm","text":"AbstractEnergyTerm modeling the objective function when there are no integral objective\n\nbeginaligned\n f(kappa)\nendaligned\n\nConstructors:     NoFETerm()     NoFETerm(:: Function)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, _obj_cell_integral, _obj_integral, _compute_gradient_k!\n\n\n\n\n\n","category":"type"},{"location":"reference/#PDENLPModels.PDENLPMeta","page":"Reference","title":"PDENLPModels.PDENLPMeta","text":"PDENLPMeta\n\nA composite type that represents the main features of the PDE-constrained optimization problem\n\n\n\nPDENLPMeta contains the following attributes:\n\ntnrj : structure representing the objective term\nYpde : TrialFESpace for the solution of the PDE\nYcon : TrialFESpace for the parameter\nXpde : TestFESpace for the solution of the PDE\nXcon : TestFESpace for the parameter\nY : concatenated TrialFESpace\nX : concatenated TestFESpace\nop : operator representing the PDE-constraint (nothing if no constraints)\nnvar_pde :number of dofs in the solution functions\nnvar_con : number of dofs in the control functions\nnparam : number of real unknowns\nnnzh_obj : number of nonzeros elements in the objective hessian\nHrows : store the structure for the hessian of the lagrangian\nHcols : store the structure for the hessian of the lagrangian\nJrows : store the structure for the hessian of the jacobian\nJcols : store the structure for the hessian of the jacobian\n\n\n\n\n\n","category":"type"},{"location":"reference/#PDENLPModels.PDEWorkspace","page":"Reference","title":"PDENLPModels.PDEWorkspace","text":"PDEWorkspace\n\nPre-allocated memory for GridapPDENLPModel.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PDENLPModels._compute_gradient!","page":"Reference","title":"PDENLPModels._compute_gradient!","text":"Return the gradient of the objective function and set it in place.\n\n_compute_gradient!(:: AbstractVector, :: EnergyFETerm, :: AbstractVector, :: FEFunctionType, :: FESpace, :: FESpace)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _obj_cell_integral, _compute_hess_k_coo\n\n\n\n\n\n","category":"function"},{"location":"reference/#PDENLPModels._compute_gradient_k!","page":"Reference","title":"PDENLPModels._compute_gradient_k!","text":"Return the derivative of the objective function w.r.t. κ.\n\n_compute_gradient_k!(::AbstractVector, :: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _obj_cell_integral, _compute_hess_k_coo\n\n\n\n\n\n","category":"function"},{"location":"reference/#PDENLPModels._compute_hess_k_vals!","page":"Reference","title":"PDENLPModels._compute_hess_k_vals!","text":"Return the values of the hessian w.r.t. κ of the objective function.\n\n_compute_hess_k_vals!(:: AbstractVector, :: AbstractNLPModel, :: AbstractEnergyTerm, :: AbstractVector, :: AbstractVector)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_integral, _obj_cell_integral, _compute_gradient_k\n\n\n\n\n\n","category":"function"},{"location":"reference/#PDENLPModels._compute_hess_structure-Tuple{AbstractEnergyTerm, Vararg{Any, 4}}","page":"Reference","title":"PDENLPModels._compute_hess_structure","text":"_compute_hess_structure(::AbstractEnergyTerm, Y, X, x0, nparam)\n_compute_hess_structure::AbstractEnergyTerm, op, Y, Ypde, Ycon, X, Xpde, x0, nparam)\n\nReturn a triplet with the structure (rows, cols) and the number of non-zeros elements in the hessian w.r.t. y and u of the objective function.\n\nThe rows and cols returned by _compute_hess_structure_obj are already shifter by nparam.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels._functions_to_vectors!-Tuple{Integer, Integer, Gridap.Geometry.Triangulation, Function, Function, Any, Gridap.FESpaces.FESpace, AbstractVector, AbstractVector}","page":"Reference","title":"PDENLPModels._functions_to_vectors!","text":"_functions_to_vectors!(nini :: Int, nfields :: Int, trian :: Triangulation, lfunc :: Function, ufunc :: Function, cell_xm :: Gridap.Arrays.AppliedArray, Y :: FESpace, lvar :: AbstractVector, uvar :: AbstractVector)\n\nIterate for k = 1 to nfields and switch lfunc[k] and ufunc[k] to vectors,  allocated in lvar and uvar in place starting from nini + 1. It returns nini + the number of allocations.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels._obj_integral","page":"Reference","title":"PDENLPModels._obj_integral","text":"Return the integral of the objective function\n\n_obj_integral(:: AbstractEnergyTerm, :: FEFunctionType, :: AbstractVector)\n\nSee also: MixedEnergyFETerm, EnergyFETerm, NoFETerm, _obj_cell_integral, _compute_gradient_k, _compute_hess_k_coo\n\n\n\n\n\n","category":"function"},{"location":"reference/#PDENLPModels._split_FEFunction-Tuple{AbstractVector, Gridap.FESpaces.FESpace, Gridap.FESpaces.FESpace}","page":"Reference","title":"PDENLPModels._split_FEFunction","text":"_split_FEFunction(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})\n\nSplit the vector x into two FEFunction corresponding to the solution y and the control u. Returns nothing for the control u if Ycon == nothing.\n\nDo not verify the compatible length.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels._split_vector-Tuple{AbstractVector, Gridap.FESpaces.FESpace, Gridap.FESpaces.FESpace}","page":"Reference","title":"PDENLPModels._split_vector","text":"_split_vector(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})\n\nSplit the vector x into three vectors: y, u, k. Returns nothing for the control u if Ycon == nothing.\n\nDo not verify the compatible length.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels._split_vectors-Tuple{AbstractVector, Gridap.FESpaces.FESpace, Gridap.FESpaces.FESpace}","page":"Reference","title":"PDENLPModels._split_vectors","text":"_split_vectors(x, Ypde, Ycon)\n\nTake a vector x and returns a splitting in terms of y, u and θ.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.bounds_functions_to_vectors-Tuple{Gridap.FESpaces.FESpace, Gridap.FESpaces.FESpace, Gridap.FESpaces.FESpace, Gridap.Geometry.Triangulation, Vararg{Function, 4}}","page":"Reference","title":"PDENLPModels.bounds_functions_to_vectors","text":"bounds_functions_to_vectors(Y :: MultiFieldFESpace, Ycon :: Union{FESpace, Nothing},  Ypde :: FESpace, trian :: Triangulation, lyfunc :: Union{Function, AbstractVector}, uyfunc :: Union{Function, AbstractVector}, lufunc :: Union{Function, AbstractVector}, uufunc :: Union{Function, AbstractVector})\n\nReturn the bounds lvar and uvar.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Hcols-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Hcols","text":"get_Hcols(nlp)\nget_Hcols(meta)\n\nReturn the value Hcols from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Hrows-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Hrows","text":"get_Hrows(nlp)\nget_Hrows(meta)\n\nReturn the value Hrows from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Jcols-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Jcols","text":"get_Jcols(nlp)\nget_Jcols(meta)\n\nReturn the value Jcols from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Jrows-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Jrows","text":"get_Jrows(nlp)\nget_Jrows(meta)\n\nReturn the value Jrows from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_X-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_X","text":"get_X(nlp)\nget_X(meta)\n\nReturn the value X from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Xcon-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Xcon","text":"get_Xcon(nlp)\nget_Xcon(meta)\n\nReturn the value Xcon from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Xpde-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Xpde","text":"get_Xpde(nlp)\nget_Xpde(meta)\n\nReturn the value Xpde from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Y-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Y","text":"get_Y(nlp)\nget_Y(meta)\n\nReturn the value Y from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Ycon-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Ycon","text":"get_Ycon(nlp)\nget_Ycon(meta)\n\nReturn the value Ycon from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_Ypde-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_Ypde","text":"get_Ypde(nlp)\nget_Ypde(meta)\n\nReturn the value Ypde from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_nnzh_obj-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_nnzh_obj","text":"get_nnzh_obj(nlp)\nget_nnzh_obj(meta)\n\nReturn the value nnzh_obj from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_nparam-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_nparam","text":"get_nparam(nlp)\nget_nparam(meta)\n\nReturn the value nparam from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_nvar_con-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_nvar_con","text":"get_nvar_con(nlp)\nget_nvar_con(meta)\n\nReturn the value nvar_con from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_nvar_pde-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_nvar_pde","text":"get_nvar_pde(nlp)\nget_nvar_pde(meta)\n\nReturn the value nvar_pde from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_op-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_op","text":"get_op(nlp)\nget_op(meta)\n\nReturn the value op from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.get_tnrj-Tuple{PDENLPModels.PDENLPMeta}","page":"Reference","title":"PDENLPModels.get_tnrj","text":"get_tnrj(nlp)\nget_tnrj(meta)\n\nReturn the value tnrj from meta or nlp.meta.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PDENLPModels.split_vectors-Tuple{GridapPDENLPModel, AbstractVector}","page":"Reference","title":"PDENLPModels.split_vectors","text":"split_vectors(::GridapPDENLPModel, x)\n\nTake a vector x and returns a splitting in terms of y, u and θ.\n\n\n\n\n\n","category":"method"},{"location":"tore/#Calculus-of-Variations","page":"Calculus of Variations","title":"Calculus of Variations","text":"","category":"section"},{"location":"tore/#The-Brachistochrone-over-a-Tore","page":"Calculus of Variations","title":"The Brachistochrone over a Tore","text":"","category":"section"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"In this example, we present how to model the classical brachistochrone problem [1] over the torus with PDENLPModels.jl in polar coordinates. We want to model the following problem:","category":"page"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"leftlbrace\nbeginaligned\nmin_varphi theta    int_0^1 a^2 dotvarphi^2 + (c + a cos(varphi))^2 dottheta^2dt\n  0 leq theta varphi leq 2pi\n  varphi(0)= 0 varphi(1)= pi\n  theta(0) = 0 theta(1) = pi\nendaligned\nright","category":"page"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"with a=1 and c=3. We also refer to [2] for the analytical solutions of this problem.","category":"page"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"  using Gridap, PDENLPModels\n\n  n = 100 #discretization size\n  domain = (0,1) \n  model = CartesianDiscreteModel(domain, n)\n      \n  labels = get_face_labeling(model)\n  add_tag_from_tags!(labels,\"diri1\",[2])\n  add_tag_from_tags!(labels,\"diri0\",[1])\n\n  x0 = zeros(2) # initial values\n  xf = π * ones(2) # final values\n  \n  order = 1\n  valuetype = Float64\n  reffe = ReferenceFE(lagrangian, valuetype, order)\n  V0 = TestFESpace(\n    model,\n    reffe;\n    conformity = :H1,\n    dirichlet_tags=[\"diri0\",\"diri1\"],\n  )\n  V1 = TestFESpace(\n    model,\n    reffe;\n    conformity = :H1,\n    dirichlet_tags=[\"diri0\",\"diri1\"],\n  )\n  \n  U0  = TrialFESpace(V0, [x0[1], xf[1]])\n  U1  = TrialFESpace(V0, [x0[2], xf[2]])\n  \n  V = MultiFieldFESpace([V0, V1])\n  U = MultiFieldFESpace([U0, U1])\n  nU0 = Gridap.FESpaces.num_free_dofs(U0)\n  nU1 = Gridap.FESpaces.num_free_dofs(U1)\n  \n  trian = Triangulation(model)\n  degree = 1\n  dΩ = Measure(trian, degree)\n\n  # The function under the integral:\n  # To use the function cos in Gridap: `operate(cos, x)` vaut cos(x)\n  # The square function is not available, so: `x*x` holds for $x^2$, \n  # and `∇(φ) ⊙ ∇(φ)` for `φ'^2`.\n  a = 1\n  c = 3\n  function f(x)\n    φ, θ = x\n    ∫(a * a * ∇(φ) ⊙ ∇(φ) + (c + a * (cos ∘ φ)) * (c + a * (cos ∘ φ)) * ∇(θ) ⊙ ∇(θ))dΩ\n  end\n\n  # boundaries\n  xmin = 0\n  xmax = 2*π\n  \n  nlp = GridapPDENLPModel(\n    zeros(nU0 + nU1),\n    f,\n    trian,\n    U,\n    V,\n    lvar = xmin * ones(nU0+nU1),\n    uvar = xmax * ones(nU0+nU1),\n  )","category":"page"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"Then, one can solve the problem with Ipopt via NLPModelsIpopt.jl and plot the solution.","category":"page"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"using NLPModelsIpopt\n\nstats = ipopt(nlp, print_level = 0)\n\nnn = Int(nlp.pdemeta.nvar_pde/2)\nφs = stats.solution[1:nn]\nθs = stats.solution[nn+1:2*nn]\n\nxs = (c .+ a * cos.(φs)) .* cos.(θs)\nys = (c .+ a * cos.(φs)) .* sin.(θs)\nzs = a * sin.(φs)\n\nL = stats.objective\n\nplotlyjs()\n\nlinspace(from, to, npoints) = range(from, stop=to, length=npoints)\n\n#plot a torus\nM = 100\nαs = linspace(0, 2π, M)\nβs = linspace(0, 2π, M)\nXs = (c .+ a * cos.(αs)) * cos.(βs)'\nYs = (c .+ a * cos.(αs)) * sin.(βs)'\nZs = (a * sin.(αs)) * ones(M)'\nplot3d(Xs, Ys, Zs, st=:surface, grid=false, c=:grays, axis=false, colorbar=false)\nplot3d!(xs, ys, zs, linewidth=4, color=:red, title=@sprintf(\"Geodesic on a Torus (length=%4.f)\", L), legend=false)","category":"page"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"(Image: Geodesic over the tore)","category":"page"},{"location":"tore/#References","page":"Calculus of Variations","title":"References","text":"","category":"section"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"[1] Weisstein, Eric W.  Brachistochrone Problem. From MathWorld–A Wolfram Web Resource.  mathworld.wolfram.com/BrachistochroneProblem.html","category":"page"},{"location":"tore/","page":"Calculus of Variations","title":"Calculus of Variations","text":"[2] Mark L. Irons The Curvature and Geodesics of the Torus (2005). torus.geodesics.pdf","category":"page"},{"location":"#PDENLPModels.jl-Documentation","page":"Introduction","title":"PDENLPModels.jl Documentation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"PDENLPModels is a Julia package that specializes the NLPModel API for modeling and discretizing optimization problems with mixed algebraic and PDE in the constraints.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"We consider optimization problems of the form: find functions (y,u) and κ ∈ ℜⁿ satisfying","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"leftlbrace\nbeginaligned\nminlimits_κyu    int_Omega f(κyu)dx \ntext st   texty solution of a PDE(κu)\n                   lcon leq c(κyu) leq ucon\n                   lvar leq (κyu)  leq uvar\nendaligned\nright","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The main challenges in modeling such a problem are to be able to discretize the domain and generate corresponding discretizations of the objective and constraints, and their evaluate derivatives with respect to all variables. We use Gridap.jl to define the domain, meshes, function spaces, and finite-element families to approximate unknowns, and to model functionals and sets of PDEs in a weak form.  PDENLPModels extends Gridap.jl's differentiation facilities to also obtain derivatives useful for optimization, i.e., first and second derivatives of the objective and constraint functions with respect to controls and finite-dimensional variables.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"After discretization of the domain Omega, the integral, and the derivatives, the resulting problem is a nonlinear optimization problem. PDENLPModels exports the GridapPDENLPModel type, an instance of an AbstractNLPModel, as defined in NLPModels.jl, which provides access to objective and constraint function values, to their first and second derivatives, and to any information that a solver might request from a model.  The role of NLPModels.jl is to define an API that users and solvers can rely on. It is the role of other packages to implement facilities that create models compliant with the NLPModels API. We refer to juliasmoothoptimizers.github.io for tutorials on the NLPModel API.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"As such, PDENLPModels offers an interface between generic PDE-constrained optimization problems and cutting-edge optimization solvers such as Artelys Knitro via NLPModelsKnitro.jl, Ipopt via NLPModelsIpopt.jl , DCISolver.jl, Percival.jl, and any solver accepting an AbstractNLPModel as input, see JuliaSmoothOptimizers.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Migot, T., Orban D., & Siqueira A. S. PDENLPModels.jl: A NLPModel API for optimization problems with PDE-constraints Journal of Open Source Software 7(80), 4736 (2022). 10.21105/joss.04736","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"] add PDENLPModels","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"The current version of PDENLPModels relies on Gridap v0.15.5.","category":"page"},{"location":"#Table-of-Contents","page":"Introduction","title":"Table of Contents","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"","category":"page"},{"location":"#Examples","page":"Introduction","title":"Examples","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"You can also check the tutorial Solve a PDE-constrained optimization problem on our site, juliasmoothoptimizers.github.io.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"We refer to the folder test/problems for more examples of problems of different types: calculus of variations, optimal control problem, PDE-constrained problems, and mixed PDE-contrained problems with both function and vector unknowns. An alternative is to visit the repository PDEOptimizationProblems that contains a collection of test problems. Without objective function, the problem reduces to a classical PDE and we refer to Gridap tutorials for examples.","category":"page"},{"location":"#References","page":"Introduction","title":"References","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Gridap.jl Badia, S., Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia. Journal of Open Source Software, 5(52), 2520.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"NLPModels.jl D. Orban, A. S. Siqueira and contributors (2020). NLPModels.jl: Data Structures for Optimization Models","category":"page"},{"location":"#Bug-reports-and-discussions","page":"Introduction","title":"Bug reports and discussions","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"If you think you found a bug, feel free to open an issue. Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"If you want to ask a question not suited for a bug report, feel free to start a discussion here. This forum is for general discussion about this repository and the JuliaSmoothOptimizers, so questions about any of our packages are welcome.","category":"page"}]
}