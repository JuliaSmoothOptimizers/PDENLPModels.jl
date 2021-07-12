```@meta
```

# JSOPDESolver

```
using Gridap, Krylov
```

Set of codes to use JSOSolvers tools to solve partial differential equations
modeled with Gridap. It contains examples of:
* using Krylov.jl to solve linear PDEs (`AffineFEOperator`)
* using Krylov.jl to solve the linear systems in the Newton-loop to solve
 nonlinear PDEs.
* more...

Pluto notebook examples can be found in the pluto folder.

## Krylov.jl to solve linear PDEs

```julia
include("header.jl")

###############################################################################
#Gridap resolution:
#This corresponds to a Poisson equation with Dirichlet and Neumann conditions
#described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/
function _poisson()
    domain = (0,1,0,1)
    n = 2^7
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)

    trian = Triangulation(model)
    degree = 2
    quad = Measure(trian,degree)

    V0 = TestFESpace(
      reffe=:Lagrangian, order=1, valuetype=Float64,
      conformity=:H1, model=model, dirichlet_tags="boundary")

    g(x) = 0.0
    Ug = TrialFESpace(V0,g)

    w(x) = 1.0
    a(u,v) = ∇(v)⊙∇(u)
    b_Ω(v) = v*w
    t_Ω = AffineFETerm(a,b_Ω,trian,quad)

    op_pde = AffineFEOperator(Ug,V0,t_Ω)
    return op_pde
end

op_pde = _poisson()

#Gridap.jl/src/FESpaces/FESolvers.jl
#Gridap.jl/src/Algebra/LinearSolvers.jl
@time ls  = KrylovSolver(minres; itmax = 150)
@time ls1 = LUSolver()
@time ls2 = BackslashSolver()

solver  = LinearFESolver(ls)
solver1 = LinearFESolver(ls1)
solver2 = LinearFESolver(ls2)

#Describe the matrix:
@test size(get_matrix(op_pde)) == (16129, 16129)
@test issparse(get_matrix(op_pde))
@test issymmetric(get_matrix(op_pde))

uh  = solve(solver, op_pde)
uh1 = solve(solver1,op_pde)
uh2 = solve(solver2,op_pde)
#Sad, that we don't have the stats back...

@time uh = solve(solver,op_pde)
x = get_free_values(uh)
@time uh1 = solve(solver1,op_pde)
x1 = get_free_values(uh1)
@time uh2 = solve(solver2,op_pde)
x2 = get_free_values(uh2)

@test norm(x  - x1, Inf) <= 1e-8
@test norm(x1 - x2, Inf) <= 1e-13
@show norm(get_matrix(op_pde)*x  - get_vector(op_pde),Inf) <= 1e-8
@test norm(get_matrix(op_pde)*x1 - get_vector(op_pde),Inf) <= 1e-15
@test norm(get_matrix(op_pde)*x2 - get_vector(op_pde),Inf) <= 1e-15
```

## Krylov.jl to solve the linear systems in the Newton-loop to solve nonlinear PDEs

```julia
include("header.jl")

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
```
