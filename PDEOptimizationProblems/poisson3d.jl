export poisson3d

"""
`poisson3d(; n :: Int = 10)`

This example represents a Poisson equation with Dirichlet boundary conditions
over the 3d-box, (0,1)^3, and we minimize the squared H_1-norm to a manufactured solution.
So, the minimal value is expected to be 0.

It is inspired from the 2nd tutorial in Gridap.jl:
https://gridap.github.io/Tutorials/stable/pages/t002_validation/
"""
function poisson3d(; n :: Int = 10)

    #For instance, the following lines build a CartesianDiscreteModel for the unit cube
    # (0,1)^3 with n cells per direction
    domain = (0,1,0,1,0,1)
    partition = (n,n,n)
    model = CartesianDiscreteModel(domain,partition)

    order = 1
    Xpde = TestFESpace(
      reffe=:Lagrangian, order=order, valuetype=Float64,
      conformity=:H1, model=model, dirichlet_tags="boundary")

    y0(x) = 0.0
    Ypde = TrialFESpace(Xpde, y0)

    Xcon = TestFESpace(
      reffe=:Lagrangian, order=order, valuetype=Float64,
      conformity=:L2, model=model)
    Ycon = TrialFESpace(Xcon)

    trian = Triangulation(model)
    degree = 2
    quad = CellQuadrature(trian,degree)

    function a(yu, v)
        y, u = yu

        ∇(v)⊙∇(y)# - v*u
    end
    b(v) = 0*v

    t_Ω = AffineFETerm(a,b,trian,quad)
    op = AffineFEOperator(MultiFieldFESpace([Ypde, Ycon]), Xpde, t_Ω)

    #h1(w) = ∇(w)⊙∇(w) + w*w
    k = 2*pi
    ubis(x) = (k^2)*sin(k*x[1])*x[2]
    ybis(x) =  sin(k*x[1]) * x[2]
    ∇ybis(x) = VectorValue(k*cos(k*x[1])*x[2], sin(k*x[1]), 0.)
    f(y,u) = (ybis - y) * (ybis - y) + (∇(y) - ∇ybis) ⊙ (∇(y) - ∇ybis) + (u-ubis) * (u-ubis)
    function f(yu)
        y,u = yu

        0.5 * f(y,u)
    end

    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, op, name = "poisson3d")

 return nlp
end
