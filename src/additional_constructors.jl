function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: NRJ,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace;
                           lvar  :: AbstractVector = - Inf * ones(T, length(x0)),
                           uvar  :: AbstractVector =   Inf * ones(T, length(x0)),
                           name  :: String = "Generic") where {T, NRJ <: AbstractEnergyTerm}

 nvar = length(x0)
 nnzh = nvar * (nvar + 1) / 2

 X, Y     = Xpde, Ypde
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = 0
 nparam   = nvar - (nvar_pde + nvar_con)

 @assert nparam ≥ 0 throw(DimensionError("x0", nvar_pde, nvar))
 @lencheck nvar x0 lvar uvar

 meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, nnzh=nnzh,
                     minimize=true, islp=false, name=name)

 return GridapPDENLPModel(meta, Counters(), tnrj, Ypde, nothing, Xpde, nothing,
                          Y, X, nothing, nvar_pde, nvar_con, nparam)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace;
                           lvar  :: AbstractVector = - Inf * ones(T, length(x0)),
                           uvar  :: AbstractVector =   Inf * ones(T, length(x0)),
                           name  :: String = "Generic") where T

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nparam   = length(x0) - nvar_pde

 tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, quad, nparam) : EnergyFETerm(f, trian, quad)

 return GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar = lvar, uvar = uvar, name = name)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: NRJ,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace,
                           c     :: FEOperator;
                           lvar  :: AbstractVector = - Inf * ones(T, length(x0)),
                           uvar  :: AbstractVector =   Inf * ones(T, length(x0)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where {T, NRJ <: AbstractEnergyTerm}

 return return GridapPDENLPModel(x0, tnrj, Ypde, nothing, Xpde, nothing, c;
                                 lvar = lvar, uvar = uvar, 
                                 name = name, lin = lin)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: NRJ,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           c     :: FEOperator;
                           lvar  :: AbstractVector = - Inf * ones(T, length(x0)),
                           uvar  :: AbstractVector =   Inf * ones(T, length(x0)),
                           lcon  :: AbstractVector = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde)),
                           ucon  :: AbstractVector = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde)),
                           y0    :: AbstractVector = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where {T, NRJ <: AbstractEnergyTerm}

 nvar = length(x0)
 ncon = length(lcon)

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 nparam   = nvar - (nvar_pde + nvar_con)

 @assert nparam>=0 throw(DimensionError("x0", nvar_pde + nvar_con, nvar))
 @lencheck nvar lvar uvar
 @lencheck ncon ucon y0

 nnzh = nvar * (nvar + 1) / 2

 if Xcon != nothing && Ycon != nothing
  _xpde = typeof(Xpde) <: MultiFieldFESpace ? Xpde : MultiFieldFESpace([Xpde])
  _xcon = typeof(Xcon) <: MultiFieldFESpace ? Xcon : MultiFieldFESpace([Xcon])
  #Handle the case where Ypde or Ycon are single field FE space(s).
  _ypde = typeof(Ypde) <: MultiFieldFESpace ? Ypde : MultiFieldFESpace([Ypde])
  _ycon = typeof(Ycon) <: MultiFieldFESpace ? Ycon : MultiFieldFESpace([Ycon])
  #Build Y (resp. X) the trial (resp. test) space of the Multi Field function [y,u]
  X     = MultiFieldFESpace(vcat(_xpde.spaces, _xcon.spaces))
  Y     = MultiFieldFESpace(vcat(_ypde.spaces, _ycon.spaces))
 elseif (Xcon == nothing) ⊻ (Ycon == nothing)
  throw(ErrorException("Error: Xcon or Ycon are both nothing or must be specified."))
 else
  _xpde = typeof(Xpde) <: MultiFieldFESpace ? Xpde : MultiFieldFESpace([Xpde])
  X = _xpde
  Y = Ypde
 end
 
 if typeof(c) <: AffineFEOperator #Here we expect ncon = nvar_pde
     nln = Int[]
     lin = 1:ncon
 else
     nln = setdiff(1:ncon, lin)
 end
 nnz_jac_k = nparam > 0 ? ncon * nparam : 0
 nnzj = count_nnz_jac(c, Y, Xpde, Ypde, Ycon) + nnz_jac_k

 meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon,
                     y0=y0, lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin,
                     nln=nln, minimize=true, islp=false, name=name)

 return GridapPDENLPModel(meta, Counters(), tnrj, Ypde, Ycon, Xpde, Xcon, Y, X,
                          c, nvar_pde, nvar_con, nparam)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           c     :: FEOperator;
                           lvar  :: AbstractVector = - Inf * ones(T, length(x0)),
                           uvar  :: AbstractVector =   Inf * ones(T, length(x0)),
                           lcon  :: AbstractVector = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde)),
                           ucon  :: AbstractVector = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde)),
                           y0    :: AbstractVector = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T

 nvar     = length(x0)
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 nparam   = nvar - (nvar_pde + nvar_con)

 tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, quad, nparam) : EnergyFETerm(f, trian, quad)

 return GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, 
                          lvar = lvar, uvar = uvar, 
                          lcon = lcon, ucon = ucon, y0 = y0, 
                          name = name, lin = lin)
end
