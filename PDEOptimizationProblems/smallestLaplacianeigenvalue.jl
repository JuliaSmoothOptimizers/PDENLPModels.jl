export smallestLaplacianeigenvalue

"""
`smallestLaplacianeigenvalue(; n :: Int = 10, args...)`

We solve the following problem:

 min_{u,z}   ∫_Ω​ |∇u|^2
 s.t.        ∫_Ω​ u^2 = 1,     for    x ∈  Ω
                u    = 0,     for    x ∈ ∂Ω

 The solution is an eigenvector of the smallest eigenvalue of the Laplacian operator,
 given by the value of the objective function.
 λ is an eigenvalue of the Laplacian if there exists u such that

 Δu + λ u = 0,   for    x ∈  Ω
        u = 0,   for    x ∈ ∂Ω

This example has been used in [Exercice 10.2.11 (p. 313)](G. Allaire, Analyse numérique et optimisation, Les éditions de Polytechnique)
and more eigenvalue problems can be found in Section 7.3.2

TODO:
- does the 1 work as it is? or should it be put in lcon, ucon?
- it is 1D for now.
"""
function smallestLaplacianeigenvalue(; n :: Int = 10, args...)

    #Domain
    domain = (0,1)
    partition = n
    model = CartesianDiscreteModel(domain,partition)

    #Definition of the spaces:
    Xpde = TestFESpace(reffe=:Lagrangian, order=1, valuetype=Float64,
                           conformity=:H1, model=model, dirichlet_tags="boundary")
    y0(x) = 0.0
    Ypde    = TrialFESpace(Xpde, y0)
    Xcon, Ycon = nothing, nothing

    #Integration machinery
    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian,degree)

    #Now we move to the optimization:
    function f(y)
        ∇(y)⋅∇(y)
    end

    #Definition of the constraint operator
    function res(y, v)
     integrate(y*y - 1, trian, quad) * v #Really not smart as it dusplicates the constraint...
    end
    t_Ω = FETerm(res,trian,quad)
    op = FEOperator(Ypde, Xpde, t_Ω)

    nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, op, name = "smallestLaplacianeigenvalue")

    return nlp
end
