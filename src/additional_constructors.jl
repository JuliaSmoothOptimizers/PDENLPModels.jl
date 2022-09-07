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

  rows, cols, nnzh = _compute_hess_structure(tnrj, Y, X, x0, nparam)

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
    rows,
    cols,
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
  kwargs...,
) where {S}
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nparam = length(x0) - nvar_pde

  tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, nparam, Ypde) : EnergyFETerm(f, trian, Ypde)

  return GridapPDENLPModel(x0, tnrj, Ypde, Xpde; kwargs...)
end

function GridapPDENLPModel(
  x0::S,
  f::Function,
  trian::Triangulation,
  Ypde::FESpace,
  Xpde::FESpace,
  c;
  kwargs...,
) where {S}
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nparam = length(x0) - nvar_pde

  tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, nparam, Ypde) : EnergyFETerm(f, trian, Ypde)

  return GridapPDENLPModel(x0, tnrj, Ypde, Xpde, c; kwargs...)
end

function GridapPDENLPModel(
  x0::S,
  tnrj::NRJ,
  Ypde::FESpace,
  Xpde::FESpace,
  c;
  lvar::S = fill!(similar(x0), -eltype(S)(Inf)),
  uvar::S = fill!(similar(x0), eltype(S)(Inf)),
  kwargs...,
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
    kwargs...,
  )
end

function GridapPDENLPModel(
  x0::S,
  tnrj::NRJ,
  Ypde::FESpace,
  Ycon::FESpace,
  Xpde::FESpace,
  Xcon::FESpace,
  c;
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

  if !(typeof(Xcon) <: VoidFESpace) && !(typeof(Ycon) <: VoidFESpace)
    _xpde = _fespace_to_multifieldfespace(Xpde)
    _xcon = _fespace_to_multifieldfespace(Xcon)
    #Handle the case where Ypde or Ycon are single field FE space(s).
    _ypde = _fespace_to_multifieldfespace(Ypde)
    _ycon = _fespace_to_multifieldfespace(Ycon)
    #Build Y (resp. X) the trial (resp. test) space of the Multi Field function [y,u]
    X = MultiFieldFESpace(vcat(_xpde.spaces, _xcon.spaces))
    Y = MultiFieldFESpace(vcat(_ypde.spaces, _ycon.spaces))
  elseif (typeof(Xcon) <: VoidFESpace) ⊻ (typeof(Ycon) <: VoidFESpace)
    throw(ErrorException("Error: Xcon or Ycon are both nothing or must be specified."))
  else
    #_xpde = _fespace_to_multifieldfespace(Xpde)
    X = Xpde #_xpde
    #_ypde = _fespace_to_multifieldfespace(Ypde)
    Y = Ypde #_ypde
  end

  op = if typeof(c) <: FEOperator
    c
  else
    FEOperator(c, Y, Xpde)
  end

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

  rows, cols, nnzh = _compute_hess_structure(tnrj, op, Y, Ypde, Ycon, X, Xpde, x0, nparam)
  _, _, nnzh_obj = _compute_hess_structure(tnrj, Y, X, x0, nparam)

  Jkrows, Jkcols, nnz_jac_k = jac_k_structure(nparam, ncon)
  Jrows, Jcols, nini = _jacobian_struct(op, x0, Y, Xpde, Ypde, Ycon)
  nnzj = nini + nnz_jac_k

  if typeof(op) <: AffineFEOperator #Here we expect ncon = nvar_pde
    if (nnz_jac_k > 0) || (nparam > 0)
      @warn "AffineFEOperator not implemented with additional constraints and algebraic variables."
    end
    lin = 1:ncon
    lin_nnzj = nnzj
    nln_nnzj = 0
  else
    lin_nnzj = 0
    nln_nnzj = nnzj
  end

  meta = NLPModelMeta{T, S}(
    nvar,
    x0 = x0,
    lvar = lvar,
    uvar = uvar,
    ncon = ncon,
    y0 = y0,
    lcon = lcon,
    ucon = ucon,
    lin_nnzj = lin_nnzj,
    nln_nnzj = nln_nnzj,
    nnzj = nnzj,
    nnzh = nnzh,
    lin = lin,
    minimize = true,
    islp = false,
    name = name,
  )

  pdemeta = PDENLPMeta{NRJ, typeof(op)}(
    tnrj,
    Ypde,
    Ycon,
    Xpde,
    Xcon,
    Y,
    X,
    op,
    nvar_pde,
    nvar_con,
    nparam,
    nnzh_obj,
    rows,
    cols,
    vcat(Jkrows, Jrows),
    vcat(Jkcols, Jcols .+ nparam),
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
  c;
  kwargs...,
) where {S}
  nvar = length(x0)
  nvar_pde = num_free_dofs(Ypde)
  nvar_con = num_free_dofs(Ycon)
  nparam = nvar - (nvar_pde + nvar_con)

  tnrj =
    nparam > 0 ? MixedEnergyFETerm(f, trian, nparam, Ypde, Ycon) :
    EnergyFETerm(f, trian, Ypde, Ycon)

  return GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c; kwargs...)
end

function GridapPDENLPModel(x0::S, f::Function, dΩ::Measure, args...; kwargs...) where {S}
  trian = dΩ.quad.trian
  return GridapPDENLPModel(x0, f, trian, args...; kwargs...)
end
