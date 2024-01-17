export GridapOptimalControl1DNLPModel
#Works for 1d state and control only
function GridapOptimalControl1DNLPModel(f, con, domain, partition; x0 = nothing, xf = nothing, umin = nothing, umax = nothing)

    model = CartesianDiscreteModel(domain,partition)
    labels = get_face_labeling(model)

    add_tag_from_tags!(labels,"diri0",[1])
    add_tag_from_tags!(labels,"diri1",[2])
    
    reffe = ReferenceFE(lagrangian, Float64, 1)
    if !isnothing(x0) && isnothing(xf)
        V0 = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["diri0"])
        U = TrialFESpace(V0,[x0])
    elseif isnothing(x0) && !isnothing(xf)
        V0 = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["diri1"])
        U = TrialFESpace(V0,[xf])
    elseif !isnothing(x0) && !isnothing(xf)
        V0 = TestFESpace(model, reffe; conformity=:H1, dirichlet_tags=["diri0", "diri1"])
        U = TrialFESpace(V0,[x0, xf])
    else
        V0 = TestFESpace(model, reffe; conformity=:H1, model=model)
        U = TrialFESpace(V0)
    end

    Vcon = TestFESpace(model, reffe, conformity=:C0)
    Ucon = TrialFESpace(Vcon)

    trian = Triangulation(model)
    degree = 1
    dΩ = Measure(trian, degree)

    function obj(y, u)
      ∫(f(y, u))dΩ
    end

    function state_equations(y, u, v)
        ∫(v * (con(y,u) - dt(y, v)))dΩ
    end
    op = FEOperator(state_equations, U, V0)

    nU, nUcon = Gridap.FESpaces.num_free_dofs(U), Gridap.FESpaces.num_free_dofs(Ucon)
    xin = zeros(nU + nUcon)

    if !isnothing(umin) && !isnothing(umax)
        lvar, uvar = bounds_functions_to_vectors(MultiFieldFESpace([U, Ucon]), Ucon, U, trian, -Inf * ones(nU), Inf * ones(nU), umin, umax)

        lvary, uvary = lvar[1:nU], uvar[1:nU]
        lvaru, uvaru = lvar[nU+1:nU+nUcon], uvar[nU+1:nU+nUcon]

        nlp = GridapPDENLPModel(xin, obj, trian, U, Ucon, V0, Vcon, op, lvary = lvary, uvary = uvary, lvaru = lvaru, uvaru = uvaru)
    else
        nlp = GridapPDENLPModel(xin, obj, trian, U, Ucon, V0, Vcon, op)
    end
    return nlp
end

export GridapOptimalControl2DNLPModel
function GridapOptimalControl2DNLPModel(f, con1, con2, domain, partition; x0 = [], xf = [], umin = nothing, umax = nothing, xmin = nothing, xmax = nothing)

    model = CartesianDiscreteModel(domain,partition)
      
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri1",[2])
    add_tag_from_tags!(labels,"diri0",[1])
  
    reffe = ReferenceFE(lagrangian, Float64, 1)
    if (length(x0) == 2) && (length(xf) == 1)
        V01 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0","diri1"])
        U1 = TrialFESpace(V01, [x0[1], xf[1]])
        V02 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0"])
        U2 = TrialFESpace(V02, [x0[2]])
    elseif (length(x0) == 2) && (length(xf) == 2)
        V01 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0","diri1"])
        U1 = TrialFESpace(V01, [x0[1], xf[1]])
        V02 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0","diri1"])
        U2 = TrialFESpace(V02, [x0[2], xf[2]])
    elseif (length(x0) == 2) && (length(xf) == 0)
        V01 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0"])
        U1 = TrialFESpace(V01, [x0[1]])
        V02 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0"])
        U2 = TrialFESpace(V02, [x0[2]])
    elseif (length(x0) == 1) && (length(xf) == 1)
        V01 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0","diri1"])
        U1 = TrialFESpace(V01, [x0[1], xf[1]])
        V02 = TestFESpace(model, reffe, conformity=:H1)
        U2 = TrialFESpace(V02)
    elseif (length(x0) == 1) && (length(xf) == 0)
        V01 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0"])
        U1 = TrialFESpace(V01, [x0[1]])
        V02 = TestFESpace(model, reffe, conformity=:H1)
        U2 = TrialFESpace(V02)
    elseif (length(x0) == 1) && (length(xf) == 2)
        V01 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0","diri1"])
        U1 = TrialFESpace(V01, [x0[1], xf[1]])
        V02 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri1"])
        U2 = TrialFESpace(V02, [xf[2]])
    elseif (length(x0) == 0) && (length(xf) == 1)
        V01 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri1"])
        U1 = TrialFESpace(V01, [xf[1]])
        V02 = TestFESpace(model, reffe, conformity=:H1)
        U2 = TrialFESpace(V02)
    elseif (length(x0) == 0) && (length(xf) == 0)
        V01 = TestFESpace(model, reffe, conformity=:H1)
        U1 = TrialFESpace(V01)
        V02 = TestFESpace(model, reffe, conformity=:H1)
        U2 = TrialFESpace(V02)
    elseif (length(x0) == 0) && (length(xf) == 2)
        V01 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri1"])
        U1 = TrialFESpace(V01, [xf[1]])
        V02 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri1"])
        U2 = TrialFESpace(V02, [xf[2]])
    end
    U = MultiFieldFESpace([U1, U2])
    V0 = MultiFieldFESpace([V01, V02])
    nU1, nU2 = Gridap.FESpaces.num_free_dofs(U1), Gridap.FESpaces.num_free_dofs(U2)
    Vcon = TestFESpace(model, reffe, conformity=:C0) #L2
    Ucon = TrialFESpace(Vcon)
  
    trian = Triangulation(model)
    degree = 1
    dΩ = Measure(trian, degree)

    function obj(y, u)
        ∫(f(y, u))dΩ
      end

    function state_equations(y, u, v)
        y₁, y₂ = y
        v₁, v₂ = v
        ∫(v₁ * con1(y₁, y₂, u) - dt(y₁, v₁) + v₂ * con2(y₁, y₂, u) - dt(y₂, v₂) )dΩ
    end
    op = FEOperator(state_equations, U, V0)

    nU, nUcon = Gridap.FESpaces.num_free_dofs(U), Gridap.FESpaces.num_free_dofs(Ucon)
  
    xin = zeros(nU + nUcon)

    if nothing in [xmin, xmax, umin, umax]
        @show "TODO: NotImplemented"
        nlp = GridapPDENLPModel(xin, obj, trian, U, Ucon, V0, Vcon, op)
    else
        #lvar, uvar = bounds_functions_to_vectors(MultiFieldFESpace([U1, U2, Ucon]), Ucon, U, trian, zeros(nU), vcat(γ*ones(nU1), ones(nU2)), umin, umax)
        lvar, uvar = bounds_functions_to_vectors(
            MultiFieldFESpace([U1, U2, Ucon]), 
            Ucon, U, trian, 
            xmin, xmax, umin, umax
        )
        lvary, uvary = lvar[1:nU], uvar[1:nU]
        lvaru, uvaru = lvar[nU+1:nU+nUcon], uvar[nU+1:nU+nUcon]

        nlp = GridapPDENLPModel(xin, obj, trian, U, Ucon, V0, Vcon, op, lvary = lvary, uvary = uvary, lvaru = lvaru, uvaru = uvaru)
    end
    return nlp
  end

  export adjoint_function_final_condition
  function adjoint_function_final_condition(domain, partition, mul, xf)
    model = CartesianDiscreteModel(domain,partition)
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri0",[1])
    add_tag_from_tags!(labels,"diri1",[2])
    reffe = ReferenceFE(lagrangian, Float64, 1)
    V0 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri1"])
    U = TrialFESpace(V0, xf)
    ph = FEFunction(U, mul)
    midpoint(xs) = sum(xs)/length(xs) #doesn't work on the last cell
    cell_xm = lazy_map(midpoint, ph.cell_dof_values)
    ans = get_free_values(Gridap.FESpaces.interpolate(cell_xm, U))
    return vcat(ans[1:end-1], xf)
  end
  
  export adjoint_function
  function adjoint_function(domain, partition, mul)
      model = CartesianDiscreteModel(domain,partition)
      labels = get_face_labeling(model)
      add_tag_from_tags!(labels,"diri0",[1])
      add_tag_from_tags!(labels,"diri1",[2])
      reffe = ReferenceFE(lagrangian, Float64, 1)
      V0 = TestFESpace(model, reffe, conformity=:H1, dirichlet_tags=["diri0", "diri1"])
      U = TrialFESpace(V0, [0., 0.])
      ph = FEFunction(U, mul)
      midpoint(xs) = sum(xs)/length(xs) #doesn't work on the last cell
      cell_xm = lazy_map(midpoint, ph.cell_dof_values)
      ans = get_free_values(Gridap.FESpaces.interpolate(cell_xm, U))
      return ans
  end