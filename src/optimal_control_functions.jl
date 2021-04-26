export bounded_2d_variational_problem

function bounded_2d_variational_problem(n, domain, x0, xf, xmin, xmax, f)

    partition = n
    model = CartesianDiscreteModel(domain, partition)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "diri1", [2])
    add_tag_from_tags!(labels, "diri0", [1])

    V0 = TestFESpace(
        reffe = :Lagrangian,
        order = 1,
        valuetype = Float64,
        conformity = :H1,
        model = model,
        dirichlet_tags = ["diri0", "diri1"],
    )
    V1 = TestFESpace(
        reffe = :Lagrangian,
        order = 1,
        valuetype = Float64,
        conformity = :H1,
        model = model,
        dirichlet_tags = ["diri0", "diri1"],
    )

    U0 = TrialFESpace(V0, [x0[1], xf[1]])
    U1 = TrialFESpace(V0, [x0[2], xf[2]])

    V = MultiFieldFESpace([V0, V1])
    U = MultiFieldFESpace([U0, U1])
    nU0 = Gridap.FESpaces.num_free_dofs(U0)
    nU1 = Gridap.FESpaces.num_free_dofs(U1)

    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian, degree)

    xin = zeros(nU0 + nU1)
    nlp = GridapPDENLPModel(
        xin,
        f,
        trian,
        quad,
        U,
        V,
        lvar = xmin * ones(nU0 + nU1),
        uvar = xmax * ones(nU0 + nU1),
    )
    return nlp
end

export GridapOptimalControlNLPModel
#Works for 1d state and control only
function GridapOptimalControlNLPModel(
    f,
    con,
    domain,
    partition;
    x0 = nothing,
    xf = nothing,
    umin = nothing,
    umax = nothing,
)

    model = CartesianDiscreteModel(domain, partition)
    labels = get_face_labeling(model)

    add_tag_from_tags!(labels, "diri0", [1])
    add_tag_from_tags!(labels, "diri1", [2])

    if !isnothing(x0) && isnothing(xf)
        V0 = TestFESpace(
            reffe = :Lagrangian,
            order = 1,
            valuetype = Float64,
            conformity = :H1,
            model = model,
            dirichlet_tags = ["diri0"],
        )
        U = TrialFESpace(V0, [x0])
    elseif isnothing(x0) && !isnothing(xf)
        V0 = TestFESpace(
            reffe = :Lagrangian,
            order = 1,
            valuetype = Float64,
            conformity = :H1,
            model = model,
            dirichlet_tags = ["diri1"],
        )
        U = TrialFESpace(V0, [xf])
    elseif !isnothing(x0) && !isnothing(xf)
        V0 = TestFESpace(
            reffe = :Lagrangian,
            order = 1,
            valuetype = Float64,
            conformity = :H1,
            model = model,
            dirichlet_tags = ["diri0", "diri1"],
        )
        U = TrialFESpace(V0, [x0, xf])
    else
        V0 = TestFESpace(
            reffe = :Lagrangian,
            order = 1,
            valuetype = Float64,
            conformity = :H1,
            model = model,
        )
        U = TrialFESpace(V0)
    end

    Vcon = TestFESpace(
        reffe = :Lagrangian,
        order = 1,
        valuetype = Float64,
        conformity = :C0,
        model = model,
    )
    Ucon = TrialFESpace(Vcon)

    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian, degree)

    function obj(xu)
        x, u = xu
        f(x, u)
    end

    @law conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u
    c(u, v) = conv(v, ∇(u)) #v⊙conv(u,∇(u))
    function res_pde_nl(xu, v)
        x, u = xu
        -c(x, v)
    end
    function state_equations(xu, v)
        x, u = xu
        v * con(x, u) #annoying to have v
    end
    t_Ω_nl = FETerm(res_pde_nl, trian, quad)
    t_Ω = FETerm(state_equations, trian, quad)
    op = FEOperator(U, V0, t_Ω_nl, t_Ω)

    nU, nUcon = Gridap.FESpaces.num_free_dofs(U), Gridap.FESpaces.num_free_dofs(Ucon)
    xin = zeros(nU + nUcon)

    if !isnothing(umin) && !isnothing(umax)
        lvar, uvar = bounds_functions_to_vectors(
            MultiFieldFESpace([U, Ucon]),
            Ucon,
            U,
            trian,
            -Inf * ones(nU),
            Inf * ones(nU),
            umin,
            umax,
        )

        lvary, uvary = lvar[1:nU], uvar[1:nU]
        lvaru, uvaru = lvar[nU+1:nU+nUcon], uvar[nU+1:nU+nUcon]

        nlp = GridapPDENLPModel(
            xin,
            obj,
            trian,
            quad,
            U,
            Ucon,
            V0,
            Vcon,
            op,
            lvary = lvary,
            uvary = uvary,
            lvaru = lvaru,
            uvaru = uvaru,
        )
    else
        nlp = GridapPDENLPModel(xin, obj, trian, quad, U, Ucon, V0, Vcon, op)
    end
    return nlp
end

export adjoint_function_final_condition
function adjoint_function_final_condition(domain, partition, mul, xf)
    model = CartesianDiscreteModel(domain, partition)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "diri0", [1])
    add_tag_from_tags!(labels, "diri1", [2])
    V0 = TestFESpace(
        reffe = :Lagrangian,
        order = 1,
        valuetype = Float64,
        conformity = :H1,
        model = model,
        dirichlet_tags = ["diri1"],
    )
    U = TrialFESpace(V0, xf)
    ph = FEFunction(U, mul)
    midpoint(xs) = sum(xs) / length(xs) #doesn't work on the last cell
    cell_xm = apply(midpoint, ph.cell_vals)
    ans = get_free_values(Gridap.FESpaces.interpolate(cell_xm, U))
    return vcat(ans[1:end-1], xf)
end

export adjoint_function
function adjoint_function(domain, partition, mul)
    model = CartesianDiscreteModel(domain, partition)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "diri0", [1])
    add_tag_from_tags!(labels, "diri1", [2])
    V0 = TestFESpace(
        reffe = :Lagrangian,
        order = 1,
        valuetype = Float64,
        conformity = :H1,
        model = model,
        dirichlet_tags = ["diri0", "diri1"],
    )
    U = TrialFESpace(V0, [0.0, 0.0])
    ph = FEFunction(U, mul)
    midpoint(xs) = sum(xs) / length(xs) #doesn't work on the last cell
    cell_xm = apply(midpoint, ph.cell_vals)
    ans = get_free_values(Gridap.FESpaces.interpolate(cell_xm, U))
    return ans
end

export bounded_2D_optimal_control
function bounded_2D_optimal_control(
    domain,
    n,
    f,
    con1,
    con2,
    umin,
    umax,
    xmin,
    xmax,
    x0,
    xf,
)

    partition = n
    model = CartesianDiscreteModel(domain, partition)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "diri1", [2])
    add_tag_from_tags!(labels, "diri0", [1])

    V01 = TestFESpace(
        reffe = :Lagrangian,
        order = 1,
        valuetype = Float64,
        conformity = :H1,
        model = model,
        dirichlet_tags = ["diri0", "diri1"],
    )
    U1 = TrialFESpace(V01, [x0[1], xf[1]])
    V02 = TestFESpace(
        reffe = :Lagrangian,
        order = 1,
        valuetype = Float64,
        conformity = :H1,
        model = model,
        dirichlet_tags = ["diri0"],
    )
    U2 = TrialFESpace(V02, [x0[2]])
    U = MultiFieldFESpace([U1, U2])
    V0 = MultiFieldFESpace([V01, V02])
    nU1, nU2 = Gridap.FESpaces.num_free_dofs(U1), Gridap.FESpaces.num_free_dofs(U2)
    Vcon = TestFESpace(
        reffe = :Lagrangian,
        order = 1,
        valuetype = Float64,
        conformity = :C0,
        model = model,
    ) #L2
    Ucon = TrialFESpace(Vcon)

    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian, degree)

    @law conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u
    c(u, v) = conv(v, ∇(u)) #v⊙conv(u,∇(u))
    function res_pde_nl(xu, v)
        x₁, x₂, u = xu
        v₁, v₂ = v
        -c(x₁, v₁) - c(x₂, v₂) + v₂ * con2(x₁, x₂, u) #issue with independent of u...
    end
    function res_pde(xu, v)
        x₁, x₂, u = xu
        v₁, v₂ = v
        v₁ * con1(x₁, x₂, u)
    end

    t_Ω_nl = FETerm(res_pde_nl, trian, quad)
    t_Ω = FETerm(res_pde, trian, quad)
    op = FEOperator(U, V0, t_Ω_nl, t_Ω)

    nU, nUcon = Gridap.FESpaces.num_free_dofs(U), Gridap.FESpaces.num_free_dofs(Ucon)

    xin = zeros(nU + nUcon)

    #lvar, uvar = bounds_functions_to_vectors(MultiFieldFESpace([U1, U2, Ucon]), Ucon, U, trian, zeros(nU), vcat(γ*ones(nU1), ones(nU2)), umin, umax)
    lvar, uvar = bounds_functions_to_vectors(
        MultiFieldFESpace([U1, U2, Ucon]),
        Ucon,
        U,
        trian,
        xmin,
        xmax,
        umin,
        umax,
    )
    lvary, uvary = lvar[1:nU], uvar[1:nU]
    lvaru, uvaru = lvar[nU+1:nU+nUcon], uvar[nU+1:nU+nUcon]

    nlp = GridapPDENLPModel(
        xin,
        f,
        trian,
        quad,
        U,
        Ucon,
        V0,
        Vcon,
        op,
        lvary = lvary,
        uvary = uvary,
        lvaru = lvaru,
        uvaru = uvaru,
    )
    return nlp
end
