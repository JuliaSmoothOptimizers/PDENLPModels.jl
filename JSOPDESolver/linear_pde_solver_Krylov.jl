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
    quad = CellQuadrature(trian,degree)

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
