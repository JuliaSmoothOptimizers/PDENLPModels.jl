export penalizedPoisson

"""
Let Ω=(0,1)^2, we solve the unconstrained optimization problem:
min_{u ∈ H_1^0}   0.5 ∫_Ω​ |∇u|^2 - w u dx
s.t.              u(x) = 0,     for    x ∈ ∂Ω
whre w(x)=1.0.

The minimizer of this problem is the solution of the Poisson equation:
 ∫_Ω​ (∇u ∇v - f*v)dx = 0, ∀ v ∈ Ω
 u = 0, x ∈ ∂Ω

This example has been used in [Exercice 10.2.4 (p. 308)](G. Allaire, Analyse numérique et optimisation, Les éditions de Polytechnique)
"""
function penalizedPoisson(; n :: Int = 8, args...)

    w(x)=1
    function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
        y = yu

        0.5 *( ∇(y)⊙∇(y) - w * y )
    end

    domain = (0,1,0,1)
    partition = (n,n)
    model = CartesianDiscreteModel(domain,partition)

    Xpde = TestFESpace(reffe=:Lagrangian, order=1, valuetype=Float64,
                           conformity=:H1, model=model, dirichlet_tags="boundary")
    y0(x) = 0.0
    Ypde    = TrialFESpace(Xpde, y0)

    trian = Triangulation(model)
    degree = 2
    quad = CellQuadrature(trian,degree)

    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Xpde, name = "penalizedPoisson")

    return nlp
end
