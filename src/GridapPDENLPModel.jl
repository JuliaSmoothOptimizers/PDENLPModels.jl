"""
PDENLPModels using Gridap.jl

https://github.com/gridap/Gridap.jl
Cite: Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia.
Journal of Open Source Software, 5(52), 2520.

Find functions (y,u): Y -> ℜⁿ x ℜⁿ and κ ∈ ℜⁿ satisfying

min      ∫_Ω​ f(y,u,κ) dΩ​
s.t.     y solution of a PDE(u,κ)=0
         lcon <= c(y,u,κ) <= ucon
         lvar <= (y,u,κ)  <= uvar

The weak formulation is then:
res((y,u),(v,q)) = ∫ v PDE(u,y,κ) + ∫ q c(y,u,κ)

where the unknown (y,u) is a MultiField see Tutorials 7 and 8 of Gridap.
https://gridap.github.io/Tutorials/stable/pages/t007_darcy/
https://gridap.github.io/Tutorials/stable/pages/t008_inc_navier_stokes/

The set Ω​ is represented here with *trian* and *quad*.

`GridapPDENLPModel(:: NLPModelMeta, :: Counters, :: Function, :: Triangulation, :: CellQuadrature, :: FESpace, :: Union{FESpace,Nothing}, :: FESpace, :: Union{FESpace,Nothing}, :: FESpace, :: Union{FESpace,Nothing}, :: Union{FEOperator, Nothing},  :: Int, :: Int, :: Int)`

TODO:
- time evolution pde problems.
- Handle the case where g and H are given
- Handle several terms in the objective function (via an FEOperator)?
- Handle finite dimension variables/unknwon parameters. Should we ask: f(yu, k) all the times or only when necessary (otw. f(yu) )
- Test the Right-hand side if op is an AffineFEOperator
- Improve Gridap.FESpaces.assemble_matrix in hess! to get directly the lower triangular?
- l.257, in hess!: sparse(LowerTriangular(hess_yu)) #there must be a better way for this
- Be more explicit on the different types of FETerm in  _from_term_to_terms!
- Right now we handle only AffineFEOperator and FEOperatorFromTerms [to be specified]
- Could we control the Dirichlet boundary condition? (like classical control of heat equations)
- Clean the tests.

- Missing: constraint ncon with num_free_dofs(Xpde)?

Example:
i) Unconstrained and no control
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde)
 GridapPDENLPModel(f, trian, quad, Ypde, Xpde)
ii) Bound constraints
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar)
 GridapPDENLPModel(f, trian, quad, Ypde, Xpde, lvar, uvar)
iii) PDE-Constrained
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c) #assuming lcon=ucon=zeros(ncon)
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, c) #assuming lcon=ucon=zeros(ncon)
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon)
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon)
iv) PDE-constraint and bounds
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c)
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c)
 GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon)
 GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon)

The following keyword arguments are available to all constructors:
- `name`: The name of the model (default: "Generic")
The following keyword arguments are available to the constructors for constrained problems:
- `lin`: An array of indexes of the linear constraints
(default: `Int[]` or 1:ncon if c is an AffineFEOperator)

The following keyword arguments are available to the constructors for constrained problems
explictly giving lcon and ucon:
- `y0`: An inital estimate to the Lagrangian multipliers (default: zeros)

We handle only two types of FEOperator: AffineFEOperator, and FEOperatorFromTerms
which is the obtained by the FEOperator constructor.
The terms supported in FEOperatorFromTerms are: FESource, NonlinearFETerm,
NonlinearFETermWithAutodiff, LinearFETerm, AffineFETerm.
"""
mutable struct GridapPDENLPModel <: AbstractNLPModel

  meta     :: NLPModelMeta

  counters :: Counters

  # For the objective function
  f        :: Function
  trian    :: Triangulation
  quad     :: CellQuadrature

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
                           f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace;
                           name  :: String = "Generic") where T <: AbstractFloat

 nvar = length(x0)
 nnzh = nvar * (nvar + 1) / 2

 X, Y     = Xpde, Ypde
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = 0
 nparam   = nvar - (nvar_pde + nvar_con)

 @assert nparam>=0 throw(DimensionError("x0", nvar_pde, nvar))

 meta = NLPModelMeta(nvar, x0=x0, nnzh=nnzh, minimize=true, islp=false, name=name)

 return GridapPDENLPModel(meta, Counters(), f, trian, quad, Ypde, nothing, Xpde, nothing, Y, X, nothing, nvar_pde, nvar_con, nparam)
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
                           f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace,
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T};
                           name  :: String = "Generic") where T <: AbstractFloat

 nvar = length(x0)
 nnzh = nvar * (nvar + 1) / 2

 X, Y     = Xpde, Ypde
 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = 0
 nparam   = nvar - (nvar_pde + nvar_con)

 @assert nparam>=0 throw(DimensionError("x0", nvar_pde, nvar))
 @lencheck nvar x0 lvar uvar

 meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, nnzh=nnzh, minimize=true, islp=false, name=name)

 return GridapPDENLPModel(meta, Counters(), f, trian, quad, Ypde, nothing, Xpde, nothing, Y, X, nothing, nvar_pde, nvar_con, nparam)
end

function GridapPDENLPModel(f     :: Function,
                           trian :: Triangulation,
                           quad  :: CellQuadrature,
                           Ypde  :: FESpace,
                           Xpde  :: FESpace,
                           lvar  :: AbstractVector{T},
                           uvar  :: AbstractVector{T};
                           name  :: String = "Generic") where T <: AbstractFloat

 x0 = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde))

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Xpde, lvar, uvar; name = name)
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
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: AbstractFloat

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(nvar_pde)
 ucon     = zeros(nvar_pde)

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon; name = name, lin = lin)
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

 return GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon; name = name, lin = lin)
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
                           lin   :: AbstractVector{<: Integer}=Int[]) where T <: AbstractFloat

 nvar = length(x0)
 ncon = length(lcon)
 @lencheck ncon ucon y0

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 nparam   = nvar - (nvar_pde + nvar_con)

 @assert nparam>=0 throw(DimensionError("x0", nvar_pde + nvar_con, nvar))

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
    nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true, islp=false, name=name)

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
  throw("Error: Xcon or Ycon are both nothing or must be specified.")
 else
  _xpde = typeof(Xpde) <: MultiFieldFESpace ? Xpde : MultiFieldFESpace([Xpde])
  X = _xpde
  Y = Ypde
 end

 return GridapPDENLPModel(meta, Counters(), f, trian, quad, Ypde, Ycon, Xpde, Xcon, Y, X, c, nvar_pde, nvar_con, nparam)
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
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: AbstractFloat

 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 x0 = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde) + nvar_con)

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, c, lcon, ucon; y0 = y0, name = name, lin = lin)
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
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: AbstractFloat

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(T, nvar_pde)
 ucon     = zeros(T, nvar_pde)

 return return GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon; name = name, lin = lin)
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
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: AbstractFloat

 nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
 lcon     = zeros(T, nvar_pde)
 ucon     = zeros(T, nvar_pde)
 y0       = fill!(similar(lcon), zero(T))

 return return GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon; y0 = y0, name = name, lin = lin)
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
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: AbstractFloat

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

 meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lcon=lcon, ucon=ucon,
    nnzj=nnzj, nnzh=nnzh, lin=lin, nln=nln, minimize=true, islp=false, name=name)

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
  throw("Error: Xcon or Ycon are both nothing or must be specified.")
 else
  _xpde = typeof(Xpde) <: MultiFieldFESpace ? Xpde : MultiFieldFESpace([Xpde])
  X = _xpde
  Y = Ypde
 end

 return GridapPDENLPModel(meta, Counters(), f, trian, quad, Ypde, Ycon, Xpde, Xcon, Y, X, c, nvar_pde, nvar_con, nparam)
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
                           lin   :: AbstractVector{<: Integer} = Int[]) where T <: AbstractFloat

 nvar_con = Ycon == nothing ? 0 : Gridap.FESpaces.num_free_dofs(Ycon)
 x0 = zeros(T, Gridap.FESpaces.num_free_dofs(Ypde) + nvar_con)

 return GridapPDENLPModel(x0, f, trian, quad, Ypde, Ycon, Xpde, Xcon, lvar, uvar, c, lcon, ucon; y0 = y0, name = name, lin = lin)
end

show_header(io :: IO, nlp :: GridapPDENLPModel) = println(io, "GridapPDENLPModel")

"""
`_split_FEFunction(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})`

Split the vector x into two FEFunction corresponding to the solution `y` and the control `u`.
Returns nothing for the control `u` if Ycon == nothing.

Do not verify the compatible length.
"""
function _split_FEFunction(x    :: AbstractVector,
                           Ypde :: FESpace,
                           Ycon :: FESpace)

    nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
    nvar_con = Gridap.FESpaces.num_free_dofs(Ycon)

    yh = FEFunction(Ypde, x[1:nvar_pde])
    uh = FEFunction(Ycon, x[1+nvar_pde:nvar_pde+nvar_con])

 return yh, uh
end

function _split_FEFunction(x    :: AbstractVector,
                           Ypde :: FESpace,
                           Ycon :: Nothing)

    nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
    yh = FEFunction(Ypde, x[1:nvar_pde])

 return yh, nothing
end

"""
`_split_vector(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})`

Split the vector x into three vectors: y, u, k.
Returns nothing for the control `u` if Ycon == nothing.

Do not verify the compatible length.
"""
function _split_vector(x    :: AbstractVector,
                       Ypde :: FESpace,
                       Ycon :: FESpace)

    nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
    nvar_con = Gridap.FESpaces.num_free_dofs(Ycon)
    #nparam   = length(x) - (nvar_pde + nvar_con)

    y = x[1:nvar_pde]
    u = x[1+nvar_pde:nvar_pde+nvar_con]
    k = x[nvar_pde+nvar_con+1:length(x)]

 return y, u, k
end

function _split_vector(x    :: AbstractVector,
                       Ypde :: FESpace,
                       Ycon :: Nothing)

    nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
    #nparam   = length(x) - nvar_pde

    y = x[1:nvar_pde]
    k = x[nvar_pde+1:length(x)]

 return y, [], k
end

function obj(nlp :: GridapPDENLPModel, x :: AbstractVector)

 @lencheck nlp.meta.nvar x
 increment!(nlp, :neval_obj)

 yu  = FEFunction(nlp.Y, x)
 if nlp.nparam > 0
   @warn "Tanj: test obj"
   κ   = x[nlp.meta.nvar - nlp.nparam+1:nlp.meta.nvar]
   int = integrate(nlp.f(yu, κ), nlp.trian, nlp.quad)
 else
   int = integrate(nlp.f(yu), nlp.trian, nlp.quad)
 end

 return sum(int)
end

function grad!(nlp :: GridapPDENLPModel, x :: AbstractVector, g :: AbstractVector)

    @lencheck nlp.meta.nvar x g
    increment!(nlp, :neval_grad)

    assem = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    yu    = FEFunction(nlp.Y, x)

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell_yu)
         yuh = CellField(nlp.Y, cell_yu)
        _yuh = Gridap.FESpaces.restrict(yuh, nlp.trian)
        integrate(nlp.f(_yuh), nlp.trian, nlp.quad)
    end

    #Compute the gradient with AD
    cell_r_yu = Gridap.Arrays.autodiff_array_gradient(_cell_obj_yu, cell_yu, cell_id_yu)
    #Put the result in the format expected by Gridap.FESpaces.assemble_matrix
    vecdata_yu = [[cell_r_yu], [cell_id_yu]]
    #Assemble the gradient in the "good" space
    g .= Gridap.FESpaces.assemble_vector(assem, vecdata_yu)

    return g
end

include("hessian_func.jl")

function hess_coo(nlp :: GridapPDENLPModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))

    yu    = FEFunction(nlp.Y, x)

    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell_yu)
         yuh = CellField(nlp.Y, cell_yu)
        _yuh = Gridap.FESpaces.restrict(yuh, nlp.trian)
        integrate(nlp.f(_yuh), nlp.trian, nlp.quad)
    end

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_obj_yu, cell_yu, cell_id_yu)
    #Assemble the matrix in the "good" space
    assem      = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    (I ,J, V) = assemble_hess(assem, cell_r_yu, cell_id_yu)

    return (I ,J, V)
end

function hess(nlp :: GridapPDENLPModel, x :: AbstractVector; obj_weight :: Real = one(eltype(x)))
    @lencheck nlp.meta.nvar x
    increment!(nlp, :neval_hess)


    (I, J, V) = hess_coo(nlp, x, obj_weight = obj_weight)

    mdofs = Gridap.FESpaces.num_free_dofs(nlp.X)
    ndofs = Gridap.FESpaces.num_free_dofs(nlp.Y)

    @assert mdofs == ndofs #otherwise there is an error in the Trial/Test spaces

    hess_yu = sparse(I, J, V, mdofs, ndofs)

    return hess_yu
end

function hess_coord!(nlp :: GridapPDENLPModel, x :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzh vals
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * obj(nlp, x)
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

function hess(nlp :: GridapPDENLPModel, x :: AbstractVector, λ :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon λ
  increment!(nlp, :neval_hess)
  ℓ(x) = obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)
  Hx = ForwardDiff.hessian(ℓ, x)
  return tril(Hx)
end

function hess_structure!(nlp :: GridapPDENLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  n = nlp.meta.nvar
  @lencheck nlp.meta.nnzh rows cols
  I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
  rows .= getindex.(I, 1)
  cols .= getindex.(I, 2)
  return rows, cols
end

function hess_coord!(nlp :: GridapPDENLPModel, x :: AbstractVector, λ :: AbstractVector, vals :: AbstractVector; obj_weight :: Real = one(eltype(x)))
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

function hprod!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
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

function hess_op!(nlp :: GridapPDENLPModel, x :: AbstractVector, Hv :: AbstractVector; obj_weight::Real=one(eltype(x)))
  @lencheck nlp.meta.nvar x Hv
  (rows, cols, vals) = hess_coo(nlp, x, obj_weight = obj_weight)
  prod = @closure v -> coo_sym_prod!(cols, rows, vals, v, Hv)
  return LinearOperator{eltype(x)}(nlp.meta.nvar, nlp.meta.nvar,
                                   true, true, prod, prod, prod)
end

function hess_op(nlp :: GridapPDENLPModel, x :: AbstractVector; obj_weight::Real=one(eltype(x)))
  @lencheck nlp.meta.nvar x
  Hv = similar(x)
  return hess_op!(nlp, x, Hv)
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

    yu = FEFunction(nlp.Y, x)
    v  = Gridap.FESpaces.get_cell_basis(nlp.Xpde) #nlp.op.test

    w, r = [], []
    for term in op.terms

     w, r = _from_term_to_terms!(term, yu, v, w, r)

    end

    assem_y = Gridap.FESpaces.SparseMatrixAssembler(nlp.Ypde, nlp.Xpde)
    Gridap.FESpaces.assemble_vector!(res, assem_y, (w,r))

    return res
end

function _from_term_to_terms!(term :: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm},
                              yu   :: Union{Gridap.FESpaces.SingleFieldFEFunction, Gridap.MultiField.MultiFieldFEFunction},
                              v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                              w    :: AbstractVector,
                              r    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yu = restrict(yu, term.trian)

 cellvals = integrate(term.res(_yu, _v), term.trian, term.quad)
 cellids  = Gridap.FESpaces.get_cell_id(term)

 Gridap.FESpaces._push_vector_contribution!(w, r, cellvals, cellids)

 return w, r
end

function _from_term_to_terms!(term :: Gridap.FESpaces.FETerm, #FESource, AffineFETerm
                              yu   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                              v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
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

function jac(nlp :: GridapPDENLPModel, x :: AbstractVector{T}) where T <: AbstractFloat
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_jac)

  pde_jac = _from_terms_to_jacobian(nlp.op, x, nlp.Y, nlp.Xpde, nlp.Ypde, nlp.Ycon)

  return pde_jac
end

function _from_terms_to_jacobian(op   :: AffineFEOperator,
                                 x    :: AbstractVector{T},
                                 Y    :: FESpace,
                                 Xpde :: FESpace,
                                 Ypde :: FESpace,
                                 Ycon :: Union{FESpace, Nothing}) where T <: AbstractFloat

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
                                 Ycon :: Union{FESpace, Nothing}) where T <: AbstractFloat

 yh, uh = _split_FEFunction(x, Ypde, Ycon)
 yu     = FEFunction(Y, x)

 dy  = Gridap.FESpaces.get_cell_basis(Ypde)
 du  = Ycon != nothing ? Gridap.FESpaces.get_cell_basis(Ycon) : nothing #use only jac is furnished
 dyu = Gridap.FESpaces.get_cell_basis(Y)
 v   = Gridap.FESpaces.get_cell_basis(Xpde) #nlp.op.test

 wu, wy  = [], []
 ru, ry  = [], []
 cu, cy  = [], []
 w, r, c = [], [], []

 for term in op.terms

   _jac_from_term_to_terms!(term, yu,  yh, uh,
                                  dyu, dy, du,
                                  v,
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
                                  yu    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  yh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  uh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction,Nothing},
                                  dyu   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  dy    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  du    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField,Nothing},
                                  v     :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
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
                                  yu    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  yh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  uh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction,Nothing},
                                  dyu   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  dy    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  du    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField,Nothing},
                                  v     :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
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
                                  yu    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  yh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  uh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction,Nothing},
                                  dyu   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  dy    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  du    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField,Nothing},
                                  v     :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
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
     _jac_from_term_to_terms_u!(term, yh, uh, du, v, wu, ru, cu)
 end

 _jac_from_term_to_terms_y!(term, yh, uh, dy, v, wy, ry, cy)

end

#https://github.com/gridap/Gridap.jl/blob/758a8620756e164ba0e6b83dc8dcbb278015b3d9/src/FESpaces/FETerms.jl#L332
function _jac_from_term_to_terms!(term  :: Gridap.FESpaces.FESource,
                                  yu    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  yh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                  uh    :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction,Nothing},
                                  dyu   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  dy    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                  du    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField,Nothing},
                                  v     :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
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
                                    yh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                    uh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                    du   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                    v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yh = restrict(yh, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)

 function uh_to_cell_residual(uf)
   _uf = Gridap.FESpaces.restrict(uf, term.trian)
   integrate(term.res(vcat(_yh, _uf), _v), term.trian, term.quad)
 end
 cellvals_u = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(uh_to_cell_residual, uh, cellids)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c, cellvals_u, cellids)

 return w, r, c
end

function _jac_from_term_to_terms_y!(term :: Gridap.FESpaces.NonlinearFETermWithAutodiff,
                                    yh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                    uh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction, Nothing},
                                    dy   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField, Nothing},
                                    v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _uh = (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true,()}}(undef,0) : restrict(uh, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)

 function yh_to_cell_residual(yf)
   _yf = Gridap.FESpaces.restrict(yf, term.trian)
   integrate(term.res(vcat(_yf,_uh),_v), term.trian, term.quad)
 end

 cellvals_y = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(yh_to_cell_residual, yh, cellids)

 Gridap.FESpaces._push_matrix_contribution!(w, r, c ,cellvals_y, cellids)

 return w, r, c
end

function _jac_from_term_to_terms_y!(term :: Gridap.FESpaces.NonlinearFETerm,
                                    yh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction},
                                    uh   :: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction, Nothing},
                                    dy   :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField, Nothing},
                                    v    :: Union{Gridap.CellData.GenericCellField, Gridap.MultiField.MultiFieldCellField},
                                    w    :: AbstractVector,
                                    r    :: AbstractVector,
                                    c    :: AbstractVector)

 _v  = restrict(v,  term.trian)
 _yh = restrict(yh, term.trian)
 _uh = (uh == nothing) ? Array{Gridap.CellData.GenericCellField{true,()}}(undef,0) : restrict(uh, term.trian)
 _dy = restrict(dy, term.trian)

 cellids  = Gridap.FESpaces.get_cell_id(term)

 cellvals_y = integrate(term.jac(vcat(_yh, _uh), _du, _v), term.trian, term.quad)

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

function jac_op(nlp :: GridapPDENLPModel, x :: AbstractVector{T}) where T <: AbstractFloat
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
                               obj_weight :: T) where T <: AbstractFloat

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
                               obj_weight :: T) where T <: AbstractFloat

 agrad(t) = ForwardDiff.gradient(x->(obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)), x + t*v)
 Hv .= ForwardDiff.derivative(t -> agrad(t), 0.)

 return Hv
end

include("hess_difficulty.jl")

include("additional_functions.jl")
