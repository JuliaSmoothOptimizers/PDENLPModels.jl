function GridapPDENLPModel(
  x0::S,
  tnrj::NRJ,
  Ypde::FESpace,
  Xpde::FESpace;
  lvar::S = fill!(similar(x0), -eltype(S)(Inf)),
  uvar::S = fill!(similar(x0), eltype(S)(Inf)),
  name::String = "Generic",
) where {S, NRJ <: AbstractEnergyTerm}
  nvar = length(x0)
  T = eltype(S)

  #_xpde = typeof(Xpde) <: MultiFieldFESpace ? Xpde : MultiFieldFESpace([Xpde])
  X = Xpde #_xpde
  #_ypde = typeof(Ypde) <: MultiFieldFESpace ? Ypde : MultiFieldFESpace([Ypde])
  Y = Ypde #_ypde
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nvar_con = 0
  nparam = nvar - (nvar_pde + nvar_con)

  @assert nparam ≥ 0 throw(DimensionError("x0", nvar_pde, nvar))

  robj, rck, cobj, cck, no, nk = _compute_hess_structure(tnrj, Y, X, x0, nparam)
  nnzh = nk + no

  if NRJ <: NoFETerm && typeof(lvar) <: AbstractVector && typeof(uvar) <: AbstractVector
    lv, uv = lvar, uvar
  else
    lv, uv = bounds_functions_to_vectors(Y, VoidFESpace(), Ypde, tnrj.trian, lvar, uvar, T[], T[])
  end

  @lencheck nvar x0 lv uv

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lv,
    uvar = uv,
    nnzh = nnzh,
    minimize = true,
    islp = false,
    name = name,
  )

  pdemeta = PDENLPMeta{NRJ, Nothing}(
    tnrj,
    Ypde,
    VoidFESpace(),
    Xpde,
    VoidFESpace(),
    Y,
    X,
    nothing,
    nvar_pde,
    nvar_con,
    nparam,
    nnzh,
    robj,
    cobj,
    rck, 
    cck,
    Int[],
    Int[],
    Int[],
    Int[],
    Int[],
    Int[],
    Int[],
    Int[],
    Int[],
    Int[],
  )

  workspace = PDEWorkspace(T, S, nvar, 0, nparam, nnzh, 0)

  return GridapPDENLPModel(meta, Counters(), pdemeta, workspace)
end

function GridapPDENLPModel(
  x0::S,
  f::Function,
  trian::Triangulation,
  Ypde::FESpace,
  Xpde::FESpace;
  lvar::S = fill!(similar(x0), -eltype(S)(Inf)),
  uvar::S = fill!(similar(x0), eltype(S)(Inf)),
  name::String = "Generic",
) where {S}
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nparam = length(x0) - nvar_pde

  tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, nparam, Ypde) : EnergyFETerm(f, trian, Ypde)

  return GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar = lvar, uvar = uvar, name = name)
end

function GridapPDENLPModel(
  x0::S,
  f::Function,
  trian::Triangulation,
  Ypde::FESpace,
  Xpde::FESpace,
  c::FEOperator;
  lvar::S = fill!(similar(x0), -eltype(S)(Inf)),
  uvar::S = fill!(similar(x0), eltype(S)(Inf)),
  name::String = "Generic",
) where {S}
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nparam = length(x0) - nvar_pde

  tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, nparam, Ypde) : EnergyFETerm(f, trian, Ypde)

  return GridapPDENLPModel(x0, tnrj, Ypde, Xpde, c, lvar = lvar, uvar = uvar, name = name)
end

function GridapPDENLPModel(
  x0::S,
  tnrj::NRJ,
  Ypde::FESpace,
  Xpde::FESpace,
  c::FEOperator;
  lvar::S = fill!(similar(x0), -eltype(S)(Inf)),
  uvar::S = fill!(similar(x0), eltype(S)(Inf)),
  name::String = "Generic",
  lin::AbstractVector{<:Integer} = Int[],
) where {S, NRJ <: AbstractEnergyTerm}
  npde = Gridap.FESpaces.num_free_dofs(Ypde)
  ndisc = length(x0) - npde

  return return GridapPDENLPModel(
    x0,
    tnrj,
    Ypde,
    VoidFESpace(),
    Xpde,
    VoidFESpace(),
    c;
    lvary = lvar[1:npde],
    uvary = uvar[1:npde],
    lvark = lvar[(npde + 1):(npde + ndisc)],
    uvark = uvar[(npde + 1):(npde + ndisc)],
    name = name,
    lin = lin,
  )
end

function GridapPDENLPModel(
  x0::S,
  tnrj::NRJ,
  Ypde::FESpace,
  Ycon::FESpace,
  Xpde::FESpace,
  Xcon::FESpace,
  c::FEOperator;
  lvary::AbstractVector = fill!(S(undef, num_free_dofs(Ypde)), -eltype(S)(Inf)),
  uvary::AbstractVector = fill!(S(undef, num_free_dofs(Ypde)), eltype(S)(Inf)),
  lvaru::AbstractVector = fill!(S(undef, num_free_dofs(Ycon)), -eltype(S)(Inf)),
  uvaru::AbstractVector = fill!(S(undef, num_free_dofs(Ycon)), eltype(S)(Inf)),
  lvark::AbstractVector = fill!(
    S(undef, max(length(x0) - num_free_dofs(Ypde) - num_free_dofs(Ycon), 0)),
    -eltype(S)(Inf),
  ),
  uvark::AbstractVector = fill!(
    S(undef, max(length(x0) - num_free_dofs(Ypde) - num_free_dofs(Ycon), 0)),
    eltype(S)(Inf),
  ),
  lcon::S = fill!(S(undef, num_free_dofs(Ypde)), zero(eltype(S))),
  ucon::S = fill!(S(undef, num_free_dofs(Ypde)), zero(eltype(S))),
  y0::S = fill!(S(undef, num_free_dofs(Ypde)), zero(eltype(S))),
  name::String = "Generic",
  lin::AbstractVector{<:Integer} = Int[],
) where {S, NRJ <: AbstractEnergyTerm}
  nvar = length(x0)
  ncon = length(lcon)
  T = eltype(S)

  nvar_pde = num_free_dofs(Ypde)
  nvar_con = num_free_dofs(Ycon)
  nparam = nvar - (nvar_pde + nvar_con)

  @assert nparam >= 0 throw(DimensionError("x0", nvar_pde + nvar_con, nvar))

  Y, X = _to_multifieldfespace(Ypde, Xpde, Ycon, Xcon)

  if NRJ == NoFETerm && typeof(lvary) <: AbstractVector && typeof(uvary) <: AbstractVector
    lvar, uvar = vcat(lvary, lvaru, lvark), vcat(uvary, uvaru, uvark)
  elseif NRJ != NoFETerm
    fun_lvar, fun_uvar =
      bounds_functions_to_vectors(Y, Ycon, Ypde, tnrj.trian, lvary, uvary, lvaru, uvaru)
    lvar, uvar = vcat(fun_lvar, lvark), vcat(fun_uvar, uvark)
  else #NRJ == FETerm and 
    #NotImplemented: NoFETerm and functional bounds
    @warn "GridapPDENLPModel: NotImplemented NoFETerm and functional bounds, ignores the functional bounds"
    #In theory can be taken from Operator but it depends which type.
    lvar, uvar = -T(Inf) * ones(T, nvar), T(Inf) * ones(T, nvar)
  end

  @lencheck nvar lvar uvar
  @lencheck ncon ucon y0

  rk, ro, rck, rc, ck, co, cck, cc, nk, no, nck, nc = _compute_hess_structure(tnrj, c, Y, Ypde, Ycon, X, Xpde, x0, nparam)
  nnzh_obj = nk + no
  nnzh = nnzh_obj + nck + nc
  #_, _, nnzh_obj = _compute_hess_structure(tnrj, Y, X, x0, nparam)

  if typeof(c) <: AffineFEOperator #Here we expect ncon = nvar_pde
    lin = 1:ncon
  end
  Jkrows, Jkcols, nnz_jac_k = jac_k_structure(nparam, ncon)
  Jyrows, Jurows, Jycols, Jucols, nini_y, nini_u = _jacobian_struct(c, x0, Y, Xpde, Ypde, Ycon)
  nnzj = nini_y + nini_u + nnz_jac_k

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    nnzj = nnzj,
    nnzh = nnzh,
    lin = lin,
    minimize = true,
    islp = false,
    name = name,
  )

  pdemeta = PDENLPMeta{NRJ, typeof(c)}(
    tnrj,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    Y,
    X,
    c,
    nvar_pde,
    nvar_con,
    nparam,
    nnzh_obj,
    rk,
    ck,
    ro,
    co,
    rck, 
    cck, 
    rc, 
    cc,
    Jkrows,
    Jkcols,
    Jyrows,
    Jycols,
    Jurows,
    Jucols,
  )

  workspace = PDEWorkspace(T, S, nvar, ncon, nparam, nnzh, nnzj)

  return GridapPDENLPModel(meta, Counters(), pdemeta, workspace)
end

function GridapPDENLPModel(
  x0::S,
  f::Function,
  trian::Triangulation,
  Ypde::FESpace,
  Ycon::FESpace,
  Xpde::FESpace,
  Xcon::FESpace,
  c::FEOperator;
  lvary::AbstractVector = fill!(S(undef, num_free_dofs(Ypde)), -eltype(S)(Inf)),
  uvary::AbstractVector = fill!(S(undef, num_free_dofs(Ypde)), eltype(S)(Inf)),
  lvaru::AbstractVector = fill!(S(undef, num_free_dofs(Ycon)), -eltype(S)(Inf)),
  uvaru::AbstractVector = fill!(S(undef, num_free_dofs(Ycon)), eltype(S)(Inf)),
  lvark::AbstractVector = fill!(
    S(undef, max(length(x0) - num_free_dofs(Ypde) - num_free_dofs(Ycon), 0)),
    -eltype(S)(Inf),
  ),
  uvark::AbstractVector = fill!(
    S(undef, max(length(x0) - num_free_dofs(Ypde) - num_free_dofs(Ycon), 0)),
    eltype(S)(Inf),
  ),
  lcon::S = fill!(S(undef, num_free_dofs(Ypde)), zero(eltype(S))),
  ucon::S = fill!(S(undef, num_free_dofs(Ypde)), zero(eltype(S))),
  y0::S = fill!(S(undef, num_free_dofs(Ypde)), zero(eltype(S))),
  name::String = "Generic",
  lin::AbstractVector{<:Integer} = Int[],
) where {S}
  nvar = length(x0)
  nvar_pde = num_free_dofs(Ypde)
  nvar_con = num_free_dofs(Ycon)
  nparam = nvar - (nvar_pde + nvar_con)

  tnrj =
    nparam > 0 ? MixedEnergyFETerm(f, trian, nparam, Ypde, Ycon) :
    EnergyFETerm(f, trian, Ypde, Ycon)

  return GridapPDENLPModel(
    x0,
    tnrj,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    c,
    lvary = lvary,
    uvary = uvary,
    lvaru = lvaru,
    uvaru = uvaru,
    lvark = lvark,
    uvark = uvark,
    lcon = lcon,
    ucon = ucon,
    y0 = y0,
    name = name,
    lin = lin,
  )
end
