export poissonmixte

"""
`poissonmixte(;args...)`

This corresponds to a Poisson equation with Dirichlet and Neumann conditions
described in the Gridap Tutorials:
https://gridap.github.io/Tutorials/stable/pages/t001_poisson/

"""
function poissonmixte(;args...)

    model = DiscreteModelFromFile(string(dirname(@__FILE__),"/models/model.json"))

    Xpde = TestFESpace(
      reffe=:Lagrangian, order=1, valuetype=Float64,
      conformity=:H1, model=model, dirichlet_tags="sides")

    g(x) = 2.0
    Ug =
    Ypde = TrialFESpace(Xpde,g)

    Xcon = TestFESpace(
            reffe=:Lagrangian, order=1, valuetype=Float64,
            conformity=:H1, model=model)
    Ycon = TrialFESpace(Xcon)

    Y  = MultiFieldFESpace([Ypde, Ycon])

    trian = Triangulation(model)
    degree = 2
    quad = CellQuadrature(trian,degree)

    neumanntags = ["circle", "triangle", "square"]
    btrian = BoundaryTriangulation(model,neumanntags)
    bquad = CellQuadrature(btrian,degree)

    ybis(x) =  x[1]^2+x[2]^2
    f(y,u) = 0.5 * (ybis - y) * (ybis - y) + 0.5 * u * u
    function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
        y, u = yu
        f(y,u)
    end

    function res_Ω(yu, v)
     y, u = yu
     v

     ∇(v)⊙∇(y) - v*u
    end
    topt_Ω = FETerm(res_Ω, trian, quad)
    h(x) = 3.0
    b_Γ(v) = v*h
    t_Γ = FESource(b_Γ,btrian,bquad)

    function res_Γ(yu, v)
     y, u = yu
     v

     -v*h
    end
    topt_Γ = FETerm(res_Γ, btrian, bquad)

    op = FEOperator(Y, Xpde, topt_Ω, topt_Γ) #FEOperator(Ypde, Xpde, topt_Ω, topt_Γ)

    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, op)

    taff_Ω = AffineFETerm(res_Ω, v->0*v, trian, quad)
    op_affine = AffineFEOperator(Y, Xpde, taff_Ω, t_Γ) #AffineFEOperator(Ypde,Xpde,taff_Ω, t_Γ)

    nlp_affine = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, op_affine)

    return nlp_affine
end
