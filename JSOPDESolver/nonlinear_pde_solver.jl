include("header.jl")

function _pdeonlyincompressibleNS()
    n = 5
    domain = (0,1,0,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri1",[6,])
    add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

    D = 2
    order = 2
    V = TestFESpace(
      reffe=:Lagrangian, conformity=:H1, valuetype=VectorValue{D,Float64},
      model=model, labels=labels, order=order, dirichlet_tags=["diri0","diri1"])

    Q = TestFESpace(
      reffe=:PLagrangian, conformity=:L2, valuetype=Float64,
      model=model, order=order-1, constraint=:zeromean)

    uD0 = VectorValue(0,0)
    uD1 = VectorValue(1,0)
    U = TrialFESpace(V,[uD0,uD1])
    P = TrialFESpace(Q)

    X = MultiFieldFESpace([V, Q])
    Y = MultiFieldFESpace([U, P])

    Re = 10.0
    @law conv(u,∇u) = Re*(∇u')⋅u
    @law dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

    function a(y,x)
      u, p = y
      v, q = x
      ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u)
    end

    c(u,v) = v⊙conv(u,∇(u))
    dc(u,du,v) = v⊙dconv(du,∇(du),u,∇(u))

    function res(y,x)
      u, p = y
      v, q = x
      a(y,x) + c(u,v)
    end

    function ja(y,dy,x)
      u, p = y
      v, q = x
      du, dp = dy
      a(dy,x)+ dc(u,du,v)
    end

    trian = Triangulation(model)
    degree = (order-1)*2
    quad = CellQuadrature(trian,degree)
    t_Ω = FETerm(res,trian,quad)
    op = FEOperator(Y,X,t_Ω)

    t_with_jac_Ω = FETerm(res,ja,trian,quad)
    op_with_jac = FEOperator(Y,X,t_with_jac_Ω)
end

op = _pdeonlyincompressibleNS()

using LineSearches
#Gridap way of solving the equation:
nls = NLSolver(
  show_trace=false, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

#struct NewtonRaphsonSolver <:NonlinearSolver
#  ls::LinearSolver
#  tol::Float64
#  max_nliters::Int
#end
nls2 = Gridap.Algebra.NewtonRaphsonSolver(LUSolver(), 1e-6, 100)
solver2 = FESolver(nls2)

#The first approach is to use Newton method anticipated by Gridap and using
#Krylov.jl to solve the linear problem.
#NLSolver(ls::LinearSolver;kwargs...)
ls  = KrylovSolver(cgls; itmax = 10000, verbose = false)
nls_krylov = NLSolver(ls, show_trace=false)
@test nls_krylov.ls == ls
solver_krylov = FESolver(nls_krylov)

nls_krylov2 = Gridap.Algebra.NewtonRaphsonSolver(ls, 1e-6, 100)
solver_krylov2 = FESolver(nls_krylov2)

#Another version is to surcharge:
#solve!(x::AbstractVector,nls::NewNonlinearSolverType,op::NonlinearOperator,cache::Nothing)

#
# Finally, we solve the problem:
#solve(solver, op)
#solve(solver2, op)
#solve(solver_krylov, op)

@time uph1 = solve(solver,op)
sol_gridap1 = get_free_values(uph1);
@time uph2 = solve(solver2,op)
sol_gridap2 = get_free_values(uph2);
@time uph3 = solve(solver_krylov,op)
sol_gridap3 = get_free_values(uph3);
@time uph4 = solve(solver_krylov2,op)
sol_gridap4 = get_free_values(uph4);

nUg = num_free_dofs(op.trial)
@test size(Gridap.FESpaces.jacobian(op, uph1)) == (nUg, nUg)

@show norm(Gridap.FESpaces.residual(op, uph1),Inf)
@show norm(Gridap.FESpaces.residual(op, uph2),Inf)
@show norm(Gridap.FESpaces.residual(op, uph3),Inf)
@show norm(Gridap.FESpaces.residual(op, uph4),Inf)

@show norm(sol_gridap1 - sol_gridap2, Inf)
@show norm(sol_gridap1 - sol_gridap3, Inf)
@show norm(sol_gridap1 - sol_gridap4, Inf)

#solve(FESolver, op): set 0 as initial guess and call:
#solve!(x,nls,op): abstract function

#Another option is to create an NLSModel.
#https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/NLSModels.jl
#using NLPModels
#nvar = Gridap.FESpaces.num_free_dofs(op.trial)
#nequ = Gridap.FESpaces.num_free_dofs(op.test)
#F(x) = Gridap.FESpaces.residual(op, FEFunction(op.trial, x))
#nls = ADNLSModel(F, zeros(nvar), nequ)

#using NLPModels: increment!
#import NLPModels: jac_residual
#function jac_residual(nls :: ADNLSModel, x :: AbstractVector)
#  @lencheck nls.meta.nvar x
#  increment!(nls, :neval_jac_residual)
#  return Gridap.FESpaces.jacobian(op, FEFunction(op.trial, x))
#end #note that this is a sparse matrix

#However, this is redundant with the GridapPDENLPModel:
#with trian, quad are nothing...
#nlp = GridapPDENLPModel(x->0., **, **, op.trial, op.test, op)

#function residual! end
#function jac_structure_residual! end
#(rows,cols) = jac_structure_residual!(nls, rows, cols)
#function jac_coord_residual! end
#vals = jac_coord_residual!(nls, x, vals)
#function jprod_residual! end
#Jv = jprod_residual!(nls, x, v, Jv)
#function jtprod_residual! end
#Jtv = jtprod_residual!(nls, x, v, Jtv)
#function hess_structure_residual! end
#hess_structure_residual!(nls, rows, cols)
#function hess_coord_residual! end
#vals = hess_coord_residual!(nls, x, v, vals)
#function hprod_residual! end
#Hiv = hprod_residual!(nls, x, i, v, Hiv)
