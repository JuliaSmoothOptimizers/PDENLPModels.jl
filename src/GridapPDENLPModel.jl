@doc raw"""
PDENLPModels using Gridap.jl

https://github.com/gridap/Gridap.jl
Cite: Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia.
Journal of Open Source Software, 5(52), 2520.

Find functions (y,u): Y -> ℜⁿ x ℜⁿ and κ ∈ ℜⁿ satisfying

min      ∫_Ω​ f(κ,y,u) dΩ​
s.t.     y solution of a PDE(κ,u)=0
         lcon <= c(κ,y,u) <= ucon
         lvar <= (κ,y,u)  <= uvar

         ```math
         \begin{aligned}
         \min_{κ,y,u} \ & ∫_Ω​ f(κ,y,u) dΩ​ \\
         \mbox{ s.t. } & y \mbox{ solution of } PDE(κ,u)=0, \\
         & lcon <= c(κ,y,u) <= ucon, \\
         & lvar <= (κ,y,u)  <= uvar.
         \end{aligned}
         ```

The weak formulation is then:
res((y,u),(v,q)) = ∫ v PDE(κ,y,u) + ∫ q c(κ,y,u)

where the unknown (y,u) is a MultiField see [Tutorials 7](https://gridap.github.io/Tutorials/stable/pages/t007_darcy/)
 and [8](https://gridap.github.io/Tutorials/stable/pages/t008_inc_navier_stokes/) of Gridap.

The set Ω​ is represented here with *trian* and *quad*.

TODO:
[ ] time evolution pde problems.   
[ ] Handle the case where g and H are given.   
[ ] Handle several terms in the objective function (via an FEOperator)?   
[ ] Be more explicit on the different types of FETerm in  _from_term_to_terms!   
[ ] Could we control the Dirichlet boundary condition? (like classical control of heat equations)   
[ ] Clean the tests.   
[ ] Missing: constraint ncon with num_free_dofs(Xpde)?   

Main constructor:

`GridapPDENLPModel(:: NLPModelMeta, :: Counters, :: AbstractEnergyTerm, :: FESpace, :: Union{FESpace,Nothing}, :: FESpace, :: Union{FESpace,Nothing}, :: FESpace, :: Union{FESpace,Nothing}, :: Union{FEOperator, Nothing}, :: Int, :: Int, :: Int)`

Additional constructors:
- Unconstrained and no control
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde)   
 GridapPDENLPModel(f, trian, quad, Ypde, Xpde)   
- Bound constraints:   
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar)   
 GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lvar, uvar)   
- PDE-constrained:   
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, c)   
 GridapPDENLPModel(Ypde, Xpde, c)   
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c)   
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, c)   
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon)   
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon)   
- PDE-constrained and bounds:   
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, c)
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c)   
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c)   
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon)   
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon)   

**Future constructors**:
- Functional bounds: in this case |lvar|=|uvar|=nparam
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lfunc, ufunc)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lfunc, ufunc)   
 GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lfunc, ufunc)
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, lfunc, ufunc)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar, lfunc, ufunc)   
 GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lvar, uvar, lfunc, ufunc) 
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, lfunc, ufunc, c)
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c)   
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c)   
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon)   
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon)     
- Discrete constraints (ck, lckon, uckon) only for problems with nparam > 0 (hence only if x0 given or tnrj)
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, ck, lckon, uckon) 
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, c, ck, lckon, uckon) 
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, ck, lckon, uckon)
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon, ck, lckon, uckon)  
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, c, ck, lckon, uckon)
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, ck, lckon, uckon) 
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon, ck, lckon, uckon) 
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lfunc, ufunc, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lfunc, ufunc, ck, lckon, uckon) 
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, lfunc, ufunc, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar, lfunc, ufunc, ck, lckon, uckon)
 GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar, lfunc, ufunc, c, ck, lckon, uckon)
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, ck, lckon, uckon)
 GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon, ck, lckon, uckon)   
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, lfunc, ufunc, c, lcon, ucon, ck, lckon, uckon)

The following keyword arguments are available to all constructors:
- `name`: The name of the model (default: "Generic")
The following keyword arguments are available to the constructors for
constrained problems:
- `lin`: An array of indexes of the linear constraints
(default: `Int[]` or 1:ncon if c is an AffineFEOperator)

The following keyword arguments are available to the constructors for
constrained problems explictly giving lcon and ucon:
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)

Notes:
 - We handle two types of FEOperator: AffineFEOperator, and FEOperatorFromTerms
 which is the obtained by the FEOperator constructor.
 The terms supported in FEOperatorFromTerms are: FESource, NonlinearFETerm,
 NonlinearFETermWithAutodiff, LinearFETerm, AffineFETerm.
 - If lcon and ucon are not given, they are assumed zeros.
 - If the type can't be deduced from the argument, it is Float64.
"""
mutable struct GridapPDENLPModel <: AbstractNLPModel

  meta     :: NLPModelMeta

  counters :: Counters

  # For the objective function
  tnrj     :: AbstractEnergyTerm

  #Gridap discretization
  Ypde     :: FESpace #TrialFESpace for the solution of the PDE
  Ycon     :: Union{FESpace, Nothing} #TrialFESpace for the parameter
  Xpde     :: FESpace #TestFESpace for the solution of the PDE
  Xcon     :: Union{FESpace, Nothing} #TestFESpace for the parameter

  Y        :: FESpace #concatenated TrialFESpace
  X        :: FESpace #concatenated TestFESpace

  op       :: Union{FEOperator, Nothing}

  nvar_pde :: Int #number of dofs in the solution functions
  nvar_con :: Int #number of dofs in the control functions
  nparam   :: Int #number of unknown parameters

end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace;
                           name  :: String = "Generic") where T <: Number

 nvar = length(x0)
 nnzh = nvar * (nvar + 1) / 2

 X, Y     = Xpde, Ypde
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = 0
 nparam   = nvar - (nvar_pde + nvar_con)

 @assert nparam ≥ 0 throw(DimensionError("x0", nvar_pde, nvar))

 meta = NLPModelMeta(nvar, x0=x0, nnzh=nnzh,
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
                           name  :: String = "Generic") where T <: Number

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nparam   = length(x0) - nvar_pde

 tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, quad, nparam) : EnergyFETerm(f, trian, quad)

 return GridapPDENLPModel(x0, tnrj, Ypde, Xpde; name = name)
end

function GridapPDENLPModel(f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace;
                           name  :: String = "Generic")

 x0 = zeros(Gridap.FESpaces.num_free_dofs(Ypde))

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde; name = name)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace,
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T};
                           name  :: String = "Generic") where T <: Number

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
                           Xpde  :: FESpace,
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T};
                           name  :: String = "Generic") where T <: Number

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nparam   = length(x0) - nvar_pde

 tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, quad, nparam) : EnergyFETerm(f, trian, quad)

 return GridapPDENLPModel(x0, tnrj, Ypde, Xpde, lvar, uvar; name = name)
end

function GridapPDENLPModel(f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace,
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T};
                           name  :: String = "Generic") where T <: Number

 x0 = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde))

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar; name = name)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace,
                           c     :: FEOperator;
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar     = length(x0)
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(nvar_pde)
 ucon     = zeros(nvar_pde)

 return GridapPDENLPModel(x0, tnrj, Ypde, nothing, Xpde, nothing, c, lcon, ucon;
                          name = name, lin = lin)
end

function GridapPDENLPModel(tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace,
                           c     :: FEOperator;
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[])

 x0 = zeros(Float64, Gridap.FESpaces.num_free_dofs(Ypde))

 return GridapPDENLPModel(x0, tnrj, Ypde, Xpde, c, name = name, lin = lin)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           c     :: FEOperator;
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar     = length(x0)
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(nvar_pde)
 ucon     = zeros(nvar_pde)

 return GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon;
                          name = name, lin = lin)
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
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(nvar_pde)
 ucon     = zeros(nvar_pde)

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon,
                          c, lcon, ucon; name = name, lin = lin)
end

function GridapPDENLPModel(f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           c     :: FEOperator;
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[])

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(nvar_pde)
 ucon     = zeros(nvar_pde)

 return GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon,
                          c, lcon, ucon; name = name, lin = lin)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           c     :: FEOperator,
                           lcon  :: AbstractVector{T},
                           ucon  :: AbstractVector{T};
                           y0    :: AbstractVector=fill!(similar(lcon), zero(T)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer}=Int[]) where T <: Number

 nvar = length(x0)
 ncon = length(lcon)
 @lencheck ncon ucon y0

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 nparam   = nvar - (nvar_pde + nvar_con)

 @assert nparam ≥ 0 throw(DimensionError("x0", nvar_pde + nvar_con, nvar))

 nnzh = nvar * (nvar + 1) / 2

 if typeof(c) <: AffineFEOperator #Here we expect ncon = nvar_pde
     nln = Int[]
     lin = 1:ncon
     nnzj = nnz(get_matrix(c))
 else
     nnzj = nvar * ncon
     nln = setdiff(1:ncon, lin)
 end

 meta = NLPModelMeta(nvar, x0=x0, ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
                     nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln,
                     minimize=true, islp=false, name=name)

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
                           c     :: FEOperator,
                           lcon  :: AbstractVector{T},
                           ucon  :: AbstractVector{T};
                           y0    :: AbstractVector=fill!(similar(lcon), zero(T)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer}=Int[]) where T <: Number

 nvar = length(x0)
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 nparam   = nvar - (nvar_pde + nvar_con)

 tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, quad, nparam) : EnergyFETerm(f, trian, quad)

 return GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon;
                          y0 = y0, name = name, lin = lin)
end

function GridapPDENLPModel(f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           c     :: FEOperator,
                           lcon  :: AbstractVector{T},
                           ucon  :: AbstractVector{T};
                           y0    :: AbstractVector = fill!(similar(lcon), zero(T)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 x0 = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde) + nvar_con)

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon,
                          c, lcon, ucon; y0 = y0, name = name, lin = lin)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace,
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T},
                           c     :: FEOperator;
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 return return GridapPDENLPModel(x0, tnrj, Ypde, nothing, Xpde, nothing, lvar, uvar,
                                 c; name = name, lin = lin)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T},
                           c     :: FEOperator;
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(T, nvar_pde)
 ucon     = zeros(T, nvar_pde)

 return return GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar,
                                 c, lcon, ucon; name = name, lin = lin)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T},
                           c     :: FEOperator;
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(T, nvar_pde)
 ucon     = zeros(T, nvar_pde)

 return return GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon,
                                 lvar, uvar, c, lcon, ucon;
                                 name = name, lin = lin)
end

function GridapPDENLPModel(f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T},
                           c     :: FEOperator;
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(T, nvar_pde)
 ucon     = zeros(T, nvar_pde)
 y0       = fill!(similar(lcon), zero(T))

 return return GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon,
                                 lvar, uvar, c, lcon, ucon;
                                 y0 = y0, name = name, lin = lin)
end

function GridapPDENLPModel(x0    :: AbstractVector{T},
                           tnrj  :: AbstractEnergyTerm,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T},
                           c     :: FEOperator,
                           lcon  :: AbstractVector{T},
                           ucon  :: AbstractVector{T};
                           y0    :: AbstractVector = fill!(similar(lcon), zero(T)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar = length(x0)
 ncon = length(lcon)

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 nparam   = nvar - (nvar_pde + nvar_con)

 @assert nparam>=0 throw(DimensionError("x0", nvar_pde + nvar_con, nvar))
 @lencheck nvar lvar uvar
 @lencheck ncon ucon y0

 nnzh = nvar * (nvar + 1) / 2

 if typeof(c) <: AffineFEOperator #Here we expect ncon = nvar_pde
     nln = Int[]
     lin = 1:ncon
     nnzj = nnz(get_matrix(c))
 else
     nnzj = nvar * ncon
     nln = setdiff(1:ncon, lin)
 end

 meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon,
                     y0=y0, lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, lin=lin,
                     nln=nln, minimize=true, islp=false, name=name)

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
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T},
                           c     :: FEOperator,
                           lcon  :: AbstractVector{T},
                           ucon  :: AbstractVector{T};
                           y0    :: AbstractVector = fill!(similar(lcon), zero(T)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar     = length(x0)
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 nparam   = nvar - (nvar_pde + nvar_con)

 tnrj = nparam > 0 ? MixedEnergyFETerm(f, trian, quad, nparam) : EnergyFETerm(f, trian, quad)

 return GridapPDENLPModel(x0, tnrj, Ypde, Ycon, Xpde, Xcon, lvar, uvar,
                          c, lcon, ucon, y0 = y0, name = name, lin = lin)
end

function GridapPDENLPModel(f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Ycon  :: Union{FESpace, Nothing},
                           Xpde  :: FESpace,
                           Xcon  :: Union{FESpace, Nothing},
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T},
                           c     :: FEOperator,
                           lcon  :: AbstractVector{T},
                           ucon  :: AbstractVector{T};
                           y0    :: AbstractVector = fill!(similar(lcon), zero(T)),
                           name  :: String = "Generic",
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: Number

 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)

 x0  = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde) + nvar_con)

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon,
                          lvar, uvar, c, lcon, ucon; y0 = y0, name = name, lin = lin)
end

show_header(io :: IO, nlp :: GridapPDENLPModel) = println(io, "GridapPDENLPModel")

function obj(nlp :: GridapPDENLPModel, x :: AbstractVector)

 @lencheck nlp.meta.nvar x
 increment!(nlp, :neval_obj)

 κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
 yu  = FEFunction(nlp.Y, xyu)

 int = _obj_integral(nlp.tnrj, κ, yu)

 return sum(int)
end

function grad!(nlp :: GridapPDENLPModel, x :: AbstractVector, g :: AbstractVector)

    @lencheck nlp.meta.nvar x g
    increment!(nlp, :neval_grad)

    κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
    yu     = FEFunction(nlp.Y, xyu)

    return _compute_gradient!(g, nlp.tnrj, κ, yu, nlp.Y, nlp.X)
end

function hess_coo(nlp :: GridapPDENLPModel, 
                  x   :: AbstractVector;
                  obj_weight :: Real = one(eltype(x)))

    κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
    yu     = FEFunction(nlp.Y, xyu)

    (I2 ,J2, V2) = _compute_hess_coo(nlp.tnrj, κ, yu, nlp.Y, nlp.X)

    if nlp.nparam > 0
        (I1, J1, V1) = _compute_hess_k_coo(nlp, nlp.tnrj, κ, xyu)
        if obj_weight == one(eltype(x))
          return (vcat(I1, I2),
                  vcat(J1, J2 .+ nlp.nparam),
                  vcat(V1, V2))
        else
          return (vcat(I1, I2),
                  vcat(J1, J2 .+ nlp.nparam),
                  vcat(obj_weight * V1, obj_weight * V2))
        end
    end

    if  obj_weight == one(eltype(x))
      return (I2 ,J2, V2)
    end

    return (I2 ,J2, obj_weight * V2)
end

"""
`hess_structure` returns the sparsity pattern of the Lagrangian Hessian 
in sparse coordinate format,
and
`hess_obj_structure` is only for the objective function hessian.
"""
function hess_yu_obj_structure(nlp :: GridapPDENLPModel)

  a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  ncells = num_cells(nlp.tnrj.trian)
  cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

  #Tanj: simplify count_hess_nnz_coo(a, cell_r_yu, cell_id_yu)
  #`Is` is never used here.
  cellidsrows = cell_id_yu
  cellidscols = cell_id_yu

  cell_rows   = Gridap.FESpaces.get_cell_dofs(a.test, cellidsrows)
  cell_cols   = Gridap.FESpaces.get_cell_dofs(a.trial, cellidscols)
  rows_cache  = Gridap.FESpaces.array_cache(cell_rows)
  cols_cache  = Gridap.FESpaces.array_cache(cell_cols)

  n   = _count_hess_entries(a.matrix_type, rows_cache, cols_cache,
                            cell_rows, cell_cols, a.strategy, nothing)

  I, J = allocate_coo_vectors_IJ(Gridap.FESpaces.get_matrix_type(a), n)

  nini = struct_hess_coo_numeric!(I, J, a, cell_id_yu)

  if n != nini
      @warn "hess_obj_structure!: Size of vals and number of assignements didn't match"
  end

  (I, J)
end

function hess_yu_obj_structure!(nlp    :: GridapPDENLPModel, 
                                rows   :: AbstractVector, 
                                cols   :: AbstractVector;
                                nfirst :: Int = 0,
                                cols_translate :: Int = 0) 

  a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  ncells = num_cells(nlp.tnrj.trian)
  cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

  nini = struct_hess_coo_numeric!(rows, cols, a, cell_id_yu, nfirst = nfirst, cols_translate = nlp.nparam)

  return nini
end

function hess_k_obj_structure(nlp :: GridapPDENLPModel)
    
    n, p = nlp.meta.nvar, nlp.nparam
    I = ((i,j) for i = 1:n, j = 1:p if j ≤ i)
    rows = getindex.(I, 1)[:]
    cols = getindex.(I, 2)[:]
    
    return rows, cols
end

function hess_k_obj_structure!(nlp :: GridapPDENLPModel, 
                               rows :: AbstractVector, 
                               cols :: AbstractVector) 
    
    n, p = nlp.meta.nvar, nlp.nparam
    nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
    I = ((i,j) for i = 1:n, j = 1:p if j ≤ i)
    rows[1:nnz_hess_k] .= getindex.(I, 1)[:]
    cols[1:nnz_hess_k] .= getindex.(I, 2)[:]
    
    return nnz_hess_k
end



function hess_obj_structure(nlp :: GridapPDENLPModel)
    
 if nlp.nparam != 0
     (I1, J1) = hess_k_obj_structure(nlp)
     (I2, J2) = hess_yu_obj_structure(nlp)
     return (vcat(I1, I2), vcat(J1, J2 .+ nlp.nparam))
 end
 
 return hess_yu_obj_structure(nlp)
end

function hess_obj_structure!(nlp  :: GridapPDENLPModel, 
                             rows :: AbstractVector, 
                             cols :: AbstractVector) 
 nvals = length(rows)
 @lencheck nvals cols
 
 nini = hess_k_obj_structure!(nlp, rows, cols)
 nini = hess_yu_obj_structure!(nlp, rows, cols, nfirst = nini, cols_translate = nlp.nparam)
 
 if nvals != nini
     @warn "hess_obj_structure!: Size of vals and number of assignements didn't match"
 end
 
 return (rows, cols)
end

function hess(nlp :: GridapPDENLPModel, 
              x   :: AbstractVector{T};
              obj_weight :: Real = one(T)) where T
              
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_hess)

    mdofs = Gridap.FESpaces.num_free_dofs(nlp.X) + nlp.nparam
    ndofs = Gridap.FESpaces.num_free_dofs(nlp.Y) + nlp.nparam

    if obj_weight == zero(T)
        (I, J) = hess_obj_structure(nlp)
        V      = zeros(T, length(J))
        return sparse(I, J, V, mdofs, ndofs)
    end

    (I, J, V) = hess_coo(nlp, x, obj_weight = obj_weight)

    @assert mdofs == ndofs #otherwise there is an error in the Trial/Test spaces

    hess_yu = sparse(I, J, V, mdofs, ndofs)

    return hess_yu
end

function hess_coord(nlp :: GridapPDENLPModel, x :: AbstractVector; obj_weight::Real=one(eltype(x)))
  @lencheck nlp.meta.nvar x
  
  #The issue here is that there is no meta specific for the obj only
  #vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
  
  a           = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  ncells      = num_cells(nlp.tnrj.trian)
  cell_id_yu  = Gridap.Arrays.IdentityVector(ncells)
  nnz_hess_yu = count_hess_nnz_coo_short(a, cell_id_yu)
  #add the nnz w.r.t. k; by default it is:
  nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2) + (nlp.meta.nvar - nlp.nparam) * nlp.nparam
  nnzh =  nnz_hess_yu + nnz_hess_k
  vals = Vector{eltype(x)}(undef, nnzh)
  
  return hess_coord!(nlp, x, vals; obj_weight=obj_weight)
end

function hess_coord!(nlp  :: GridapPDENLPModel,
                     x    :: AbstractVector,
                     vals :: AbstractVector;
                     obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  #@lencheck nlp.meta.nnzh vals #we trust the length of vals
  #increment!(nlp, :neval_hess)
  
  a          = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  ncells     = num_cells(nlp.tnrj.trian)
  cell_id_yu = Gridap.Arrays.IdentityVector(ncells)
  
  κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
  yu     = FEFunction(nlp.Y, xyu)
  
  nvals  = length(vals)
  
  #Right now V1 cannot be computed separately
  nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2) + (nlp.meta.nvar - nlp.nparam) * nlp.nparam
  vals[1:nnz_hess_k] .= _compute_hess_k_vals(nlp, nlp.tnrj, κ, xyu)
  
  cell_yu    = Gridap.FESpaces.get_cell_values(yu)

  function _cell_obj_yu(cell)
       yuh = CellField(nlp.Y, cell)
      _obj_cell_integral(nlp.tnrj, κ, yuh)
  end

  #Compute the hessian with AD
  cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_obj_yu, cell_yu, cell_id_yu)
  #length(nini) + length(V1) should be length(vals)
  nini = vals_hess_coo_numeric!(vals, a, cell_r_yu, cell_id_yu, nfirst = nnz_hess_k)
  
  if nvals != nini
      @warn "hess_coord!: Size of vals and number of assignements didn't match"
  end
  return vals
end

function hprod!(nlp :: GridapPDENLPModel,
                x   :: AbstractVector,
                v   :: AbstractVector,
                Hv  :: AbstractVector;
                obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  if obj_weight == zero(eltype(x))
      Hv .= zero(similar(x))
      return Hv
  end

  #Only one lower triangular of the Hessian
  (rows, cols, vals) = hess_coo(nlp, x, obj_weight = obj_weight)

  coo_sym_prod!(cols, rows, vals, v, Hv)

 return Hv
end

function hess_op!(nlp :: GridapPDENLPModel,
                  x   :: AbstractVector,
                  Hv  :: AbstractVector;
                  obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x Hv
  (rows, cols, vals) = hess_coo(nlp, x, obj_weight = obj_weight)
  prod = @closure v -> coo_sym_prod!(cols, rows, vals, v, Hv)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

function hess_op(nlp :: GridapPDENLPModel, 
                 x   :: AbstractVector; 
                 obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  Hv = similar(x)
  return hess_op!(nlp, x, Hv, obj_weight = obj_weight)
end

function hess(nlp :: GridapPDENLPModel,
              x   :: AbstractVector,
              λ   :: AbstractVector;
              obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon λ
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess_structure!(nlp :: GridapPDENLPModel,
                        rows :: AbstractVector{<: Integer},
                        cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  @lencheck nlp.meta.nnzh rows cols
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j) 
  rows .= getindex.(I, 1)[:]
  cols .= getindex.(I, 2)[:]
  return rows, cols
end

function hess_coord!(nlp  :: GridapPDENLPModel,
                     x    :: AbstractVector,
                     λ    :: AbstractVector,
                     vals :: AbstractVector;
                     obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon λ
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)
  Hx = ForwardDiff.hessian(ℓ, x)
  k = 1
  for j = 1 : nlp.meta.nvar
    for i = j : nlp.meta.nvar
      vals[k] = Hx[i, j]
      k += 1
    end
  end
  return vals
end

function cons!(nlp :: GridapPDENLPModel, x :: AbstractVector, c :: AbstractVector)

    #pde_residual = Array{eltype(x),1}(undef, nlp.nvar_pde)

    _from_terms_to_residual!(nlp.op, x, nlp, c)

    #c .= pde_residual

    return c
end

function _from_terms_to_residual!(op  :: Gridap.FESpaces.FEOperatorFromTerms,
                                  x   :: AbstractVector,
                                  nlp :: GridapPDENLPModel,
                                  res :: AbstractVector)

    κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
    yu     = FEFunction(nlp.Y, xyu)
    v      = Gridap.FESpaces.get_cell_basis(nlp.Xpde)

    w, r = [], []
    for term in op.terms

     w, r = _from_term_to_terms!(term, κ, yu, v, w, r)

    end

    assem_y = Gridap.FESpaces.SparseMatrixAssembler(nlp.Ypde, nlp.Xpde)
    Gridap.FESpaces.assemble_vector!(res, assem_y, (w,r))

    return res
end

function _from_term_to_terms!(term :: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
                              κ    :: AbstractVector,
                              yu   :: FEFunctionType,
                              v    :: CellFieldType,
                              w    :: AbstractVector,
                              r    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yu = restrict(yu, term.trian)

 if length(κ) > 0
     cellvals = integrate(term.res(κ, _yu, _v), term.trian, term.quad)
 else
     cellvals = integrate(term.res(_yu, _v), term.trian, term.quad)
 end
 cellids  = Gridap.FESpaces.get_cell_id(term)

 Gridap.FESpaces._push_vector_contribution!(w, r, cellvals, cellids)

 return w, r
end

function _from_term_to_terms!(term :: Gridap.FESpaces.FETerm, #FESource, AffineFETerm
                              κ    :: AbstractVector,
                              yu   :: FEFunctionType,
                              v    :: CellFieldType,
                              w    :: AbstractVector,
                              r    :: AbstractVector)

 cellvals = Gridap.FESpaces.get_cell_residual(term, yu, v)
 cellids  = Gridap.FESpaces.get_cell_id(term)

 Gridap.FESpaces._push_vector_contribution!(w, r, cellvals, cellids)

 return w, r
end

"""
Note:
- mul! seems faster than doing:
rows, cols, vals = findnz(get_matrix(op))
coo_prod!(cols, rows, vals, v, res)

- get_matrix(op) is a sparse matrix
- Benchmark equivalent to Gridap.FESpaces.residual!(res, op_affine.op, xrand)
"""
function _from_terms_to_residual!(op  :: AffineFEOperator,
                                  x   :: AbstractVector,
                                  nlp :: GridapPDENLPModel,
                                  res :: AbstractVector)

 T = eltype(x)
 mul!(res, get_matrix(op), x)
 axpy!(-one(T), get_vector(op), res)

 return res
end

function jac(nlp :: GridapPDENLPModel, x :: AbstractVector{T}) where T <: Number
  nvar = nlp.meta.nvar
  @lencheck nvar x
  increment!(nlp, :neval_jac)

  pde_jac = _from_terms_to_jacobian(nlp.op, x, nlp.Y, nlp.Xpde, nlp.Ypde, nlp.Ycon)

  if nlp.nparam > 0
      κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
      ck = @closure k -> cons(nlp, vcat(k, xyu))
      jac_k = ForwardDiff.jacobian(ck, κ)
      return hcat(jac_k, pde_jac)
  end

  return pde_jac
end

function _from_terms_to_jacobian(op   :: AffineFEOperator,
                                 x    :: AbstractVector{T},
                                 Y    :: FESpace,
                                 Xpde :: FESpace,
                                 Ypde :: FESpace,
                                 Ycon :: Union{FESpace, Nothing}) where T <: Number

 return get_matrix(op)
end

"""
Note:
1) Compute the derivatives w.r.t. y and u separately.

2) Use AD for those derivatives. Only for the following:
- NonlinearFETerm (we neglect the inapropriate jac function);
- NonlinearFETermWithAutodiff
- TODO: Gridap.FESpaces.FETerm & AffineFETerm ?
- FESource <: AffineFETerm (jacobian of a FESource is nothing)
- LinearFETerm <: AffineFETerm (not implemented)
"""
function _from_terms_to_jacobian(op   :: Gridap.FESpaces.FEOperatorFromTerms,
                                 x    :: AbstractVector{T},
                                 Y    :: FESpace,
                                 Xpde :: FESpace,
                                 Ypde :: FESpace,
                                 Ycon :: Union{FESpace, Nothing}) where T <: Number

 nvar   = length(x)
 nyu    = num_free_dofs(Y)
 nparam = nvar - nyu
 yh, uh = _split_FEFunction(x, Ypde, Ycon)
 κ, xyu = x[1 : nparam], x[nparam + 1 : nvar]
 yu     = FEFunction(Y, xyu)

 dy  = Gridap.FESpaces.get_cell_basis(Ypde)
 du  = Ycon != nothing ? Gridap.FESpaces.get_cell_basis(Ycon) : nothing #use only jac is furnished
 dyu = Gridap.FESpaces.get_cell_basis(Y)
 v   = Gridap.FESpaces.get_cell_basis(Xpde)

 wu, wy  = [], []
 ru, ry  = [], []
 cu, cy  = [], []
 w, r, c = [], [], []

 for term in op.terms

   _jac_from_term_to_terms!(term, κ, yu, yh, uh,
                                  dyu, dy, du, v,
                                  w,  r,  c,
                                  wu, ru, cu,
                                  wy, ry, cy)

 end

 if Ycon != nothing
     assem_u = Gridap.FESpaces.SparseMatrixAssembler(Ycon, Xpde)
     Au      = Gridap.FESpaces.assemble_matrix(assem_u, (wu, ru, cu))
 else
     Au      = zeros(Gridap.FESpaces.num_free_dofs(Ypde), 0)
 end

 assem_y = Gridap.FESpaces.SparseMatrixAssembler(Ypde, Xpde)
 Ay      = Gridap.FESpaces.assemble_matrix(assem_y, (wy, ry, cy))

 S = hcat(Ay,Au)

 assem = Gridap.FESpaces.SparseMatrixAssembler(Y, Xpde)
 #doesn't work as we may not have the good sparsity pattern.
 #Gridap.FESpaces.assemble_matrix_add!(S, assem, (w, r, c))
 S += Gridap.FESpaces.assemble_matrix(assem, (w, r, c))

 return S
end

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/SparseMatrixAssemblers.jl#L242
import Gridap.FESpaces._get_block_layout
#Tanj: this is an error when the jacobian matrix are of size 1xn.
#unit test: poinsson-with-Neumann-and-Dirichlet, l. 160.
function _get_block_layout(a::AbstractArray)
  nothing
end

function _jac_from_term_to_terms!(term  :: Gridap.FESpaces.FETerm,
                                  κ     :: AbstractVector,
                                  yu    :: FEFunctionType,
                                  yh    :: FEFunctionType,
                                  uh    :: Union{FEFunctionType,Nothing},
                                  dyu   :: CellFieldType,
                                  dy    :: CellFieldType,
                                  du    :: Union{CellFieldType,Nothing},
                                  v     :: CellFieldType,
                                  w     :: AbstractVector,
                                  r     :: AbstractVector,
                                  c     :: AbstractVector,
                                  wu    :: AbstractVector,
                                  ru    :: AbstractVector,
                                  cu    :: AbstractVector,
                                  wy    :: AbstractVector,
                                  ry    :: AbstractVector,
                                  cy    :: AbstractVector)

 @warn "_jac_from_term_to_terms!(::FETerm, ...): If that works, good for you."

 cellvals = get_cell_jacobian(term, yu, dyu, v)
 cellids  = get_cell_id(term)
 _push_matrix_contribution!(w, r, c, cellvals, cellids)

end

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/FETerms.jl#L367
function _jac_from_term_to_terms!(term  :: Union{Gridap.FESpaces.LinearFETerm, Gridap.FESpaces.AffineFETermFromIntegration},
                                  κ     :: AbstractVector,
                                  yu    :: FEFunctionType,
                                  yh    :: FEFunctionType,
                                  uh    :: Union{FEFunctionType,Nothing},
                                  dyu   :: CellFieldType,
                                  dy    :: CellFieldType,
                                  du    :: Union{CellFieldType,Nothing},
                                  v     :: CellFieldType,
                                  w     :: AbstractVector,
                                  r     :: AbstractVector,
                                  c     :: AbstractVector,
                                  wu    :: AbstractVector,
                                  ru    :: AbstractVector,
                                  cu    :: AbstractVector,
                                  wy    :: AbstractVector,
                                  ry    :: AbstractVector,
                                  cy    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yuh = restrict(yu, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)
 cellvals = integrate(term.biform(_yuh, _v), term.trian, term.quad)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals, cellids)

end

function _jac_from_term_to_terms!(term  :: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff,Gridap.FESpaces.NonlinearFETerm},
                                  κ     :: AbstractVector,
                                  yu    :: FEFunctionType,
                                  yh    :: FEFunctionType,
                                  uh    :: Union{FEFunctionType,Nothing},
                                  dyu   :: CellFieldType,
                                  dy    :: CellFieldType,
                                  du    :: Union{CellFieldType,Nothing},
                                  v     :: CellFieldType,
                                  w     :: AbstractVector,
                                  r     :: AbstractVector,
                                  c     :: AbstractVector,
                                  wu    :: AbstractVector,
                                  ru    :: AbstractVector,
                                  cu    :: AbstractVector,
                                  wy    :: AbstractVector,
                                  ry    :: AbstractVector,
                                  cy    :: AbstractVector)

 if typeof(term) == Gridap.FESpaces.NonlinearFETerm
     @warn "_jac_from_term_to_terms!: For NonlinearFETerm, function jac is used to compute the derivative w.r.t. y."
 end

 if du != nothing
     _jac_from_term_to_terms_u!(term, κ, yh, uh, du, v, wu, ru, cu)
 end

 _jac_from_term_to_terms_y!(term, κ, yh, uh, dy, v, wy, ry, cy)

end

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/FETerms.jl#L332
function _jac_from_term_to_terms!(term  :: Gridap.FESpaces.FESource,
                                  κ     :: AbstractVector,
                                  yu    :: FEFunctionType,
                                  yh    :: FEFunctionType,
                                  uh    :: Union{FEFunctionType,Nothing},
                                  dyu   :: CellFieldType,
                                  dy    :: CellFieldType,
                                  du    :: Union{CellFieldType,Nothing},
                                  v     :: CellFieldType,
                                  w     :: AbstractVector,
                                  r     :: AbstractVector,
                                  c     :: AbstractVector,
                                  wu    :: AbstractVector,
                                  ru    :: AbstractVector,
                                  cu    :: AbstractVector,
                                  wy    :: AbstractVector,
                                  ry    :: AbstractVector,
                                  cy    :: AbstractVector)

 nothing
end

function _jac_from_term_to_terms_u!(term :: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff,Gridap.FESpaces.NonlinearFETerm},
                                    κ    :: AbstractVector,
                                    yh   :: FEFunctionType,
                                    uh   :: FEFunctionType,
                                    du   :: CellFieldType,
                                    v    :: CellFieldType,
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yh = restrict(yh, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)
 function uh_to_cell_residual(uf)
   _uf = Gridap.FESpaces.restrict(uf, term.trian)
   if length(κ) > 0
     return integrate(term.res(κ, vcat(_yh, _uf), _v), term.trian, term.quad)
   else
     return integrate(term.res(vcat(_yh, _uf), _v), term.trian, term.quad)
   end
 end

 cellvals_u = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(uh_to_cell_residual, uh, cellids)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals_u, cellids)

 return w, r, c
end

function _jac_from_term_to_terms_y!(term :: Gridap.FESpaces.NonlinearFETermWithAutodiff,
                                    κ    :: AbstractVector,
                                    yh   :: FEFunctionType,
                                    uh   :: Union{FEFunctionType, Nothing},
                                    dy   :: Union{CellFieldType, Nothing},
                                    v    :: CellFieldType,
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 #_uh = (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true,()}}(undef,0) : restrict(uh, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)
 #=
 if length(κ) > 0 && uh != nothing
     _uh = restrict(uh, term.trian)
     function yh_to_cell_residual(yf)
       _yf = Gridap.FESpaces.restrict(yf, term.trian)
       integrate(term.res(vcat(_yf,_uh), κ, _v), term.trian, term.quad)
     end
 elseif length(κ) > 0 #&& uh == nothing
     function yh_to_cell_residual(yf)
       _yf = Gridap.FESpaces.restrict(yf, term.trian)
       integrate(term.res(_yf, κ,_v), term.trian, term.quad)
     end
 elseif length(κ) == 0 && uh == nothing
    function yh_to_cell_residual(yf)
      _yf = Gridap.FESpaces.restrict(yf, term.trian)
      integrate(term.res(_yf,_v), term.trian, term.quad)
    end
 else #length(κ) == 0 && uh == nothing
     _uh = restrict(uh, term.trian)
     function yh_to_cell_residual(yf)
       _yf = Gridap.FESpaces.restrict(yf, term.trian)
       integrate(term.res(vcat(_yf,_uh),_v), term.trian, term.quad)
     end
 end
 =#
 _uh = (uh != nothing) ? restrict(uh, term.trian) : nothing
 function yh_to_cell_residual(yf) #Tanj: improved solution is to declare the function outside
     _yf = Gridap.FESpaces.restrict(yf, term.trian)
     if length(κ) > 0 && uh != nothing
           return integrate(term.res(κ, vcat(_yf, _uh), _v), term.trian, term.quad)
     elseif length(κ) > 0 #&& uh == nothing
           return integrate(term.res(κ, _yf, _v), term.trian, term.quad)
     elseif length(κ) == 0 && uh == nothing
          return integrate(term.res(_yf,_v), term.trian, term.quad)
     else #length(κ) == 0 && uh == nothing
           return integrate(term.res(vcat(_yf,_uh),_v), term.trian, term.quad)
     end
 end

 cellvals_y = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(yh_to_cell_residual, yh, cellids)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c ,cellvals_y, cellids)

 return w, r, c
end

function _jac_from_term_to_terms_y!(term :: Gridap.FESpaces.NonlinearFETerm,
                                    κ    :: AbstractVector,
                                    yh   :: FEFunctionType,
                                    uh   :: Union{FEFunctionType, Nothing},
                                    dy   :: Union{CellFieldType, Nothing},
                                    v    :: CellFieldType,
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yh = restrict(yh, term.trian)
 _uh = (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true,()}}(undef,0) : restrict(uh, term.trian)
 _dy = restrict(dy, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)
 if length(κ) > 0
     cellvals_y = integrate(term.jac(κ, vcat(_yh, _uh), _du, _v), term.trian, term.quad)
 else
     cellvals_y = integrate(term.jac(vcat(_yh, _uh), _du, _v), term.trian, term.quad)
 end

 Gridap.FESpaces._push_matrix_contribution!(w, r, c ,cellvals_y, cellids)

 return w, r, c
end

"""
    Jv = jprod!(nlp, x, v, Jv)
Evaluate ``J(x)v``, the Jacobian-vector product at `x` in place.

Note for GridapPDENLPModel:
- Evaluate the jacobian and then use mul! (here coo_prod! is slower as we have to compute findnz).
- Alternative: benefit from the AD? Jv .= ForwardDiff.derivative(t->nlp.c(nlp, x + t * v), 0)
when the jacobian is obtained by AD.
"""
function jprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Jv :: AbstractVector)
  @lencheck nlp.meta.nvar x v
  @lencheck nlp.meta.ncon Jv
  increment!(nlp, :neval_jprod)

  Jx = jac(nlp, x)
  mul!(Jv, Jx, v)

  return Jv
end

"""
    Jv = jtprod!(nlp, x, v, Jv)
Evaluate ``J(x)'v``, the Jacobian-vector product at `x` in place.

Note for GridapPDENLPModel:
- Evaluate the jacobian and then use mul! (here coo_prod! is slower as we have to compute findnz).
- Alternative: benefit from the AD? Jtv .= ForwardDiff.gradient(x -> dot(nlp.c(x), v), x)
when the jacobian is obtained by AD.
"""
function jtprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon v
  increment!(nlp, :neval_jtprod)

  Jx = jac(nlp, x)
  mul!(Jtv, Jx', v)

  return Jtv
end

function jac_op(nlp :: GridapPDENLPModel, x :: AbstractVector{T}) where T <: Number
  @lencheck nlp.meta.nvar x

  Jv  = Array{T,1}(undef, nlp.meta.ncon)
  Jtv = Array{T,1}(undef, nlp.meta.nvar)

  return jac_op!(nlp, x, Jv, Jtv)
end

function jac_op!(nlp :: GridapPDENLPModel,
                 x   :: AbstractVector,
                 Jv  :: AbstractVector,
                 Jtv :: AbstractVector)
  @lencheck nlp.meta.nvar x Jtv
  @lencheck nlp.meta.ncon Jv

  Jx = jac(nlp, x)

  prod   = @closure v -> mul!(Jv,  Jx,  v)
  ctprod = @closure v -> mul!(Jtv, Jx', v)
  return LinearOperator{eltype(x)}(nlp.meta.ncon, nlp.meta.nvar,
                                   false, false, prod, ctprod, ctprod)
end

function jac_structure!(nlp :: GridapPDENLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nlp.meta.nnzj rows cols
  return _jac_structure!(nlp.op, nlp, rows, cols)
end

function _jac_structure!(op :: AffineFEOperator, nlp :: GridapPDENLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})

 #In this case, the jacobian matrix is constant:
 I, J, V = findnz(get_matrix(op))
 rows .= I
 cols .= J

 return rows, cols
end

function _jac_structure!(op :: Gridap.FESpaces.FEOperatorFromTerms, nlp :: GridapPDENLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})

 m, n = nlp.meta.ncon, nlp.meta.nvar
 I = ((i,j) for i = 1:m, j = 1:n)
 rows .= getindex.(I, 1)[:]
 cols .= getindex.(I, 2)[:]

 return rows, cols
end

function jac_coord!(nlp :: GridapPDENLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  return _jac_coord!(nlp.op, nlp, x, vals)
end

function _jac_coord!(op :: AffineFEOperator, nlp :: GridapPDENLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  I, J, V = findnz(get_matrix(op))
  vals .= V
  return vals
end

function _jac_coord!(op :: Gridap.FESpaces.FEOperatorFromTerms, nlp :: GridapPDENLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  Jx = jac(nlp, x)
  vals .= Jx[:]
  return vals
end

function hprod!(nlp  :: GridapPDENLPModel, x :: AbstractVector, λ :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))

  @lencheck nlp.meta.nvar x v Hv
  @lencheck nlp.meta.ncon λ
  increment!(nlp, :neval_hprod)

  λ_pde = λ[1:nlp.nvar_pde]
  λ_con = λ[nlp.nvar_pde + 1 : nlp.meta.ncon]

  _from_terms_to_hprod!(nlp.op, x, λ_pde, v, nlp, Hv, obj_weight)

  return Hv
end

function _from_terms_to_hprod!(op  :: Gridap.FESpaces.AffineFEOperator,
                               x   :: AbstractVector{T},
                               λ   :: AbstractVector{T},
                               v   :: AbstractVector{T},
                               nlp :: GridapPDENLPModel,
                               Hv  :: AbstractVector{T},
                               obj_weight :: T) where T <: Number

 decrement!(nlp, :neval_hprod) #otherwise we would count 2 hprod!
 #By definition the hessian of an AffineFEOperator vanishes.
 return hprod!(nlp, x, v, Hv; obj_weight = obj_weight)
end

function _from_terms_to_hprod!(op  :: Gridap.FESpaces.FEOperatorFromTerms,
                               x   :: AbstractVector{T},
                               λ   :: AbstractVector,
                               v   :: AbstractVector{T},
                               nlp :: GridapPDENLPModel,
                               Hv  :: AbstractVector{T},
                               obj_weight :: T) where T <: Number

 agrad(t) = ForwardDiff.gradient(x->(obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)), x + t*v)
 Hv .= ForwardDiff.derivative(t -> agrad(t), 0.)

 return Hv
end

include("hess_difficulty.jl")

include("additional_functions.jl")
