export Burger1d

"""

`Burger1d(;n :: Int = 512, kwargs...)`

Let Ω=(0,1), we solve the one-dimensional ODE-constrained control problem:
min_{y,u}   0.5 ∫_Ω​ |y(x) - y_d(x)|^2dx + 0.5 * α * ∫_Ω​ |u|^2
s.t.          -ν y'' + yy' = u + h,   for    x ∈  Ω,
                  u(0) = 0, u(1)=-1,  for    x ∈ ∂Ω,
where the constraint is a 1D stationary Burger's equation over Ω, with
h(x)=2(ν + x^3) and ν=0.08. The first objective measures deviation from the
data u_d(x)=-x^2, while the second term regularizes the control with α = 0.01.

This example has been used in [Section 9.1](Estrin, R., Friedlander, M. P., Orban, D., & Saunders, M. A. (2020).
Implementing a smooth exact penalty function for equality-constrained nonlinear optimization.
SIAM Journal on Scientific Computing, 42(3), A1809-A1835.)

The specificity of the problem:
- quadratic objective function;
- nonlinear constraints with AD jacobian;

Suggestions:
- FEOperatorFromTerms has only one term. We might consider splitting linear and
nonlinear terms.
"""
function Burger1d(;n :: Int = 512, kwargs...)

    #Domain
    domain = (0,1)
    partition = n
    model = CartesianDiscreteModel(domain,partition)

    #Definition of the spaces:
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri1",[2])
    add_tag_from_tags!(labels,"diri0",[1])

    Xpde = TestFESpace(
      reffe=:Lagrangian, conformity=:H1, valuetype=Float64,
      model=model, labels=labels, order=1, dirichlet_tags=["diri0","diri1"])

    uD0 = VectorValue(0)
    uD1 = VectorValue(-1)
    Ypde = TrialFESpace(Xpde,[uD0,uD1])

    Xcon = TestFESpace(
            reffe=:Lagrangian, order=1, valuetype=Float64,
            conformity=:H1, model=model)
    Ycon = TrialFESpace(Xcon)

    #Integration machinery
    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian,degree)

    #Now we move to the optimization:
    ud(x) = -x[1]^2
    α = 1e-2
    #objective function:
    f(u, z) = 0.5 * (ud - u) * (ud - u) + 0.5 * α * z * z
    function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
        u, z = yu
        f(u,z)
    end

    #Definition of the constraint operator
    h(x) = 2*(nu + x[1]^3)
    @law conv(u,∇u) = (∇u ⋅one(∇u))⊙u
    c(u,v) = v⊙conv(u,∇(u))
    nu = 0.08
    function res(yu, v)
     y, u = yu
     v

     -nu*(∇(v)⊙∇(y)) + c(y,v) - v * u - v * h
    end
    t_Ω = FETerm(res,trian,quad)
    op = FEOperator(Ypde, Xpde, t_Ω) # or FEOperator(Y, Xpde, t_Ω)

    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, op, name = "Burger1d")

    return nlp
end
