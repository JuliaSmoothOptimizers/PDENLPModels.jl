export poissonBoltzman2d

"""

`poissonBoltzman2d(; n :: Int = 100)`

Let Ω=(-1,1)^2, we solve the 2-dimensional PDE-constrained control problem:
min_{y ∈ H_1^0, u ∈ L^∞}   0.5 ∫_Ω​ |y(x) - yd(x)|^2dx + 0.5 * α * ∫_Ω​ |u|^2
 s.t.         -Δy + sinh(y) = h + u,   for    x ∈  Ω
                         y(x) = 0,     for    x ∈ ∂Ω

The force term here is h(x_1,x_2) = - sin( ω x_1)sin( ω x_2) with  ω = π - 1/8.
The targeted function is
yd(x) = {10 if x ∈ [0.25,0.75]^2, 5 otherwise}.
We discretize using P1 finite elements on a uniform mesh with 10201 triangles,
resulting in a problem with n = 20002 variables and m = 9801 constraints.
We use y_0=1 and u_0 = 1 as the initial point.

This example has been used in [Section 9.3](Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
SIAM Journal on Scientific Computing, 42(3), A1809-A1835.)

The specificity of the problem:
- quadratic objective function;
- nonlinear constraints with AD jacobian;
"""
function poissonBoltzman2d(; n :: Int = 100)

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
    yd(x) = min(x[1]-0.25, 0.75-x[1],x[2]-0.25, 0.75-x[2])>=0. ? 10. : 5.
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

     #∇(v)⊙∇(y) + sinh(y)*v - u*v - v * h
     ∇(v)⊙∇(y) + operate(sinh, y)*v - u*v - v * h
     #operate(tanh,ph)
    end
    t_Ω = FETerm(res,trian,quad)
    op = FEOperator(Y, Xpde, t_Ω)

    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, op, name = "poissonBoltzman2d")

    return nlp
end
