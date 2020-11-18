export incompressibleNavierStokes

"""

`incompressibleNavierStokes(; n :: Int64 = 3, kargs...)`

This corresponds to the incompressible Navier-Stokes equation
described in the Gridap Tutorials:
https://gridap.github.io/Tutorials/stable/pages/t008_inc_navier_stokes/

It has no objective function and no control, just the PDE.
"""
function incompressibleNavierStokes(; n :: Int64 = 3, kargs...)

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

    function jac(y,dy,x)
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

    t_with_jac_Ω = FETerm(res,jac,trian,quad)
    op_with_jac = FEOperator(Y,X,t_with_jac_Ω)

    nlp = GridapPDENLPModel(x->0.0, trian, quad, Y, nothing, X, nothing, op, name = "incompressibleNavierStokes")

    return nlp
end
