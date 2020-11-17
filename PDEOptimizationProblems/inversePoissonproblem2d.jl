# Using Gridap and GridapPDENLPModel, we solve the following
# distributed Poisson control proble with Dirichlet boundary:
#
# min_{u,z}   0.5 ∫_Ω​ |u(x) - ud(x)|^2dx + 0.5 * α * ∫_Ω​ |z|^2
# s.t.        -∇⋅(z∇u) = h ,   for    x ∈  Ω
#             u(x) = 0,        for    x ∈ ∂Ω
export inversePoissonproblem2d

"""

`inversePoissonproblem2d(;n :: Int = 512, kwargs...)`

Let Ω=(-1,1)^2, we solve the 2-dimensional PDE-constrained control problem:
min_{y ∈ H_1^0, u ∈ L^∞}   0.5 ∫_Ω​ |y(x) - y_d(x)|^2dx + 0.5 * α * ∫_Ω​ |u|^2
s.t.          -∇⋅(z∇u) = h,   for    x ∈  Ω,
                  u(x) = 0,   for    x ∈ ∂Ω.
Let c = (0.2,0.2) and and define S_1 = {x | ||x-c||_2 ≤ 0.3 } and S_2 = {x | ||x-c||_1 ≤ 0.6 }.
The target u_d is generated as the solution of the PDE with
z_*(x) = 1 + 0.5 * I_{S_1}(x) + 0.5 * I_{S_2}(x).
The force term here is h(x_1,x_2) = - sin( ω x_1)sin( ω x_2) with  ω = π - 1/8.
The control variable z represents the diffusion coefficients for the Poisson problem that we are trying to recover.
Set α = 10^{-4} and discretize using P1 finite elements on a uniform mesh of 1089
triangles and employ an identical discretization for the optimization variables u, thus n_con = 1089 and n_pde = 961.
Initial point is y_0=1 and u_0 = 1.
z ≥ 0 (implicit)

This example has been used in [Section 9.2](Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
SIAM Journal on Scientific Computing, 42(3), A1809-A1835.)

The specificity of the problem:
- quadratic objective function;
- nonlinear constraints with AD jacobian;

Suggestions/TODO:
- compute target function u_d.
- L∞ constraint
- Verify the weak formulation
"""
function inversePoissonproblem2d(; n :: Int = 100)

    #Domain
    domain = (-1,1,-1,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)

    #Definition of the spaces:
    Xpde = TestFESpace(reffe=:Lagrangian, conformity=:H1, valuetype=Float64, model=model, order=2, dirichlet_tags="boundary")

    y0(x) = 0.0
    Ypde    = TrialFESpace(Xpde, y0)

    Xcon = TestFESpace(
            reffe=:Lagrangian, order=1, valuetype=Float64,
            conformity=:H1, model=model) #should be L\infty not H1 here
    Ycon = TrialFESpace(Xcon)
    Y = MultiFieldFESpace([Ypde, Ycon])

    #Integration machinery
    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian,degree)

    #Objective function:
    yd(x) = -x[1]^2
    α = 1e-4
    f(y, u) = 0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u
    function f(yu)
        y, u = yu
        f(y,u)
    end

    #Definition of the constraint operator
    ω = π - 1/8
    h(x) = - sin( ω*x[1])*sin( ω*x[2])
    function res(yu, v)
     y, u = yu
     v

     -u*(∇(v)⊙∇(y)) - v * h
    end
    t_Ω = FETerm(res,trian,quad)
    op = FEOperator(Y, Xpde, t_Ω)

    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, op, name = "inversePoissonproblem2d")

 return nlp
end
