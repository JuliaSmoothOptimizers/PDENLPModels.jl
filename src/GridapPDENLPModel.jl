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
mutable struct GridapPDENLPModel{NRJ <: AbstractEnergyTerm} <: AbstractNLPModel

  meta     :: NLPModelMeta

  counters :: Counters

  # For the objective function
  tnrj     :: NRJ

  #Gridap discretization
  Ypde     :: FESpace #TrialFESpace for the solution of the PDE
  Ycon     :: FESpace #TrialFESpace for the parameter
  Xpde     :: FESpace #TestFESpace for the solution of the PDE
  Xcon     :: FESpace #TestFESpace for the parameter

  Y        :: FESpace #concatenated TrialFESpace
  X        :: FESpace #concatenated TestFESpace

  op       :: Union{FEOperator, Nothing}

  nvar_pde :: Int #number of dofs in the solution functions
  nvar_con :: Int #number of dofs in the control functions
  nparam   :: Int #number of unknown parameters

end

include("bounds_function.jl")
include("additional_constructors.jl")

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
      return (vcat(I1, I2 .+ nlp.nparam),
              vcat(J1, J2 .+ nlp.nparam),
              vcat(V1, V2))
    else
      return (vcat(I1, I2 .+ nlp.nparam),
              vcat(J1, J2 .+ nlp.nparam),
              vcat(obj_weight * V1, obj_weight * V2))
    end
  end

  if  obj_weight == one(eltype(x))
    return (I2, J2, V2)
  end

  return (I2, J2, obj_weight * V2)
end

function hess_yu_obj_structure(nlp :: GridapPDENLPModel)
    
  # Special case as nlp.tnrj has no field trian.    
  if typeof(nlp.tnrj) <: NoFETerm
    T = eltype(nlp.meta.nvar)
    return (T[], T[])
  end

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
                                nfirst :: Int = 0) 

  # Special case as nlp.tnrj has no field trian.    
  if typeof(nlp.tnrj) <: NoFETerm
    return nfirst
  end

  a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  ncells = num_cells(nlp.tnrj.trian)
  cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

  nini = struct_hess_coo_numeric!(rows, cols, a, cell_id_yu, nfirst = nfirst, cols_translate = nlp.nparam, rows_translate = nlp.nparam)

  return nini
end

function hess_k_obj_structure(nlp :: GridapPDENLPModel)
    
  p = nlp.nparam
  if (typeof(nlp.tnrj) <: MixedEnergyFETerm && nlp.tnrj.inde) || typeof(nlp.tnrj) <: NoFETerm
    n = nlp.nparam
  else
    n = nlp.meta.nvar
  end
  I = ((i,j) for i = 1:n, j = 1:p if j ≤ i)
  rows = getindex.(I, 1)[:]
  cols = getindex.(I, 2)[:]
    
  return rows, cols
end

function hess_k_obj_structure!(nlp  :: GridapPDENLPModel, 
                               rows :: AbstractVector, 
                               cols :: AbstractVector) 
    
  p = nlp.nparam
  if (typeof(nlp.tnrj) <: MixedEnergyFETerm && nlp.tnrj.inde) || typeof(nlp.tnrj) <: NoFETerm
    n = nlp.nparam
  else
    n = nlp.meta.nvar
  end
  nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
  I = ((i,j) for i = 1:n, j = 1:p if j ≤ i)
  rows[1:nnz_hess_k] .= getindex.(I, 1)[:]
  cols[1:nnz_hess_k] .= getindex.(I, 2)[:]
    
  return nnz_hess_k
end

"""
`hess_structure` returns the sparsity pattern of the Lagrangian Hessian 
in sparse coordinate format,
and
`hess_obj_structure` is only for the objective function hessian.
"""
function hess_obj_structure(nlp :: GridapPDENLPModel)
    
  if nlp.nparam != 0
    (I1, J1) = hess_k_obj_structure(nlp)
    (I2, J2) = hess_yu_obj_structure(nlp)
    return (vcat(I1, I2 .+ nlp.nparam), vcat(J1, J2 .+ nlp.nparam))
  end
 
  return hess_yu_obj_structure(nlp)
end

function hess_obj_structure!(nlp  :: GridapPDENLPModel, 
                             rows :: AbstractVector, 
                             cols :: AbstractVector) 
  nvals = length(rows)
  @lencheck nvals cols

  nini = hess_k_obj_structure!(nlp, rows, cols)
  nini = hess_yu_obj_structure!(nlp, rows, cols, nfirst = nini)
 
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
  #########################################################################################
  #The issue here is that there is no meta specific for the obj only
  #vals = Vector{eltype(x)}(undef, nlp.meta.nnzh)
  
  # Special case as nlp.tnrj has no field trian.    
  if typeof(nlp.tnrj) <: NoFETerm
    nnz_hess_yu = 0
  else
    a           = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    ncells      = num_cells(nlp.tnrj.trian)
    cell_id_yu  = Gridap.Arrays.IdentityVector(ncells)
    nnz_hess_yu = count_hess_nnz_coo_short(a, cell_id_yu)
  end

  #add the nnz w.r.t. k; by default it is:
  if (typeof(nlp.tnrj) <: MixedEnergyFETerm && nlp.tnrj.inde) || typeof(nlp.tnrj) <: NoFETerm
    nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2)
  else
    nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2) + (nlp.meta.nvar - nlp.nparam) * nlp.nparam
  end
  
  nnzh =  nnz_hess_yu + nnz_hess_k
  #USE get_nnzh here
  #########################################################################################
  vals = Vector{eltype(x)}(undef, nnzh)
  
  return hess_coord!(nlp, x, vals; obj_weight=obj_weight)
end

function hess_coord!(nlp  :: GridapPDENLPModel,
                     x    :: AbstractVector,
                     vals :: AbstractVector;
                     obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  #@lencheck nlp.meta.nnzh vals #we trust the length of vals
  increment!(nlp, :neval_hess)
  
  κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
  yu     = FEFunction(nlp.Y, xyu)
  
  nvals  = length(vals)
  
  #Right now V1 cannot be computed separately
  if (typeof(nlp.tnrj) <: MixedEnergyFETerm && nlp.tnrj.inde) || typeof(nlp.tnrj) <: NoFETerm
    nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2)
  else
    nnz_hess_k = Int(nlp.nparam * (nlp.nparam + 1) / 2) + (nlp.meta.nvar - nlp.nparam) * nlp.nparam
  end
  nini = nnz_hess_k
  vals[1:nnz_hess_k] .= _compute_hess_k_vals(nlp, nlp.tnrj, κ, xyu)
  
  if typeof(nlp.tnrj) != NoFETerm
    a          = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    ncells     = num_cells(nlp.tnrj.trian)
    cell_id_yu = Gridap.Arrays.IdentityVector(ncells)
    cell_yu    = Gridap.FESpaces.get_cell_values(yu)

    function _cell_obj_yu(cell)
      yuh = CellField(nlp.Y, cell)
      _obj_cell_integral(nlp.tnrj, κ, yuh)
    end

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_obj_yu, cell_yu, cell_id_yu)
    #length(nini) + length(V1) should be length(vals)
    nini = vals_hess_coo_numeric!(vals, a, cell_r_yu, cell_id_yu, nfirst = nini)
  end
  
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

#######################################################################
# TO BE REMOVED
function hess2(nlp :: GridapPDENLPModel,
              x   :: AbstractVector,
              λ   :: AbstractVector;
              obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon λ
  increment!(nlp, :neval_hess)
  function ℓ(x, λ)

    κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
    yu  = FEFunction(nlp.Y, xyu)
    int = _obj_integral(nlp.tnrj, κ, yu)
    
    c = similar(x, nlp.meta.ncon)
    _from_terms_to_residual!(nlp.op, x, nlp, c)

    return obj_weight * sum(int) + dot(c, λ)
  end

  #ℓ(x) = obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)
  Hx = ForwardDiff.hessian(x->ℓ(x, λ), x)
  return tril(Hx)
end
###########################################################################
#### Tanj: IS THIS FUNCTION NECESSARY ????
function hess(nlp :: GridapPDENLPModel, 
              x   :: AbstractVector{T},
              λ   :: AbstractVector{T};
              obj_weight :: Real = one(T)) where T
              
  @lencheck nlp.meta.nvar x
  increment!(nlp, :neval_hess)

  mdofs = Gridap.FESpaces.num_free_dofs(nlp.X) + nlp.nparam
  ndofs = Gridap.FESpaces.num_free_dofs(nlp.Y) + nlp.nparam

  if obj_weight == zero(T)
    (I, J) = hess_obj_structure(nlp)
     V     = zeros(T, length(J))
  else
    (I, J, V) = hess_coo(nlp, x, obj_weight = obj_weight)
  end

  @assert mdofs == ndofs #otherwise there is an error in the Trial/Test spaces

  (I2, J2, V2) = hess_coo(nlp, nlp.op, x, λ)
  hess_lag = sparse(vcat(I, I2), vcat(J, J2), vcat(V, V2), mdofs, ndofs)

  return hess_lag
end

"""
`hess_coo`: return the hessian of the constraints in COO-format.

Notes:    
- `hess_coo(nlp, op :: AffineFEOperator, x, λ)`: return 0-matrix
- `hess_coo(nlp, op :: FEOperatorFromTerms, x, λ)`: iterate over the terms

TODO:
make it a real COO-format function.

do not work with parameters.
"""
function hess_coo(nlp :: GridapPDENLPModel, 
                  op  :: AffineFEOperator,
                  x   :: AbstractVector,
                  λ   :: AbstractVector)
  mdofs = Gridap.FESpaces.num_free_dofs(nlp.X) + nlp.nparam
  ndofs = Gridap.FESpaces.num_free_dofs(nlp.Y) + nlp.nparam
  return findnz(spzeros(mdofs, ndofs))
end

function hess_coo(nlp :: GridapPDENLPModel, 
                  op  :: Gridap.FESpaces.FEOperatorFromTerms,
                  x   :: AbstractVector{T},
                  λ   :: AbstractVector) where T

  nnzh_obj = get_nnzh(nlp.tnrj, nlp.Y, nlp.X, nlp.nparam, nlp.meta.nvar)
  nnzh = nlp.meta.nnzh - nnzh_obj
  (rows, cols, vals) = Vector{T}(undef, nnzh), Vector{T}(undef, nnzh), Vector{T}(undef, nnzh)

  κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
  yu     = FEFunction(nlp.Y, xyu)

  cell_yu    = Gridap.FESpaces.get_cell_values(yu)
  cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

  nini = 0
  for term in op.terms
    if !(typeof(term) <: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm})
      continue
    end

    λf      = FEFunction(nlp.Xpde, λ)
    cell_λf = Gridap.FESpaces.get_cell_values(λf)
    lfh     = CellField(nlp.Xpde, cell_λf)
    _lfh    = Gridap.FESpaces.restrict(lfh, term.trian) #This is where the term play a first role.

    function _cell_res_yu(cell)
      xfh  = CellField(nlp.Y, cell)
      _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
      
      if length(κ) > 0
        _res = integrate(term.res(κ, _xfh,_lfh), term.trian, term.quad)
      else
        _res = integrate(term.res(_xfh,_lfh), term.trian, term.quad)
      end
      lag = _res
      return lag
    end

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_res_yu, cell_yu, cell_id_yu)
    #Assemble the matrix in the "good" space
    assem      = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    (I, J, V)  = assemble_hess(assem, cell_r_yu, cell_id_yu)
    nn = length(V)
    rows[nini+1:nini+nn] .= I
    cols[nini+1:nini+nn] .= J
    vals[nini+1:nini+nn] .= V

    #= What about extra parameters????
    if nlp.nparam > 0
      (I1, J1, V1) = _compute_hess_k_coo(nlp, nlp.tnrj, κ, xyu)
      if obj_weight == one(eltype(x))
        return (vcat(I1, I2 .+ nlp.nparam),
                vcat(J1, J2 .+ nlp.nparam),
                vcat(V1, V2))
      else
        return (vcat(I1, I2 .+ nlp.nparam),
                vcat(J1, J2 .+ nlp.nparam),
                vcat(obj_weight * V1, obj_weight * V2))
      end
    end
    =#
  end

  return (rows, cols, vals)
end

function hess_structure!(nlp :: GridapPDENLPModel, 
                         rows :: AbstractVector{<: Integer}, 
                         cols :: AbstractVector{<: Integer})

  #nnzh_obj = get_nnzh(nlp.tnrj, nlp.Ypde, nlp.Xpde, nparam, nvar)
  #hess_obj_structure!(nlp, @view rows[1:nnzh_obj], @view cols[1:nnzh_obj])
  #################################################""
  n, p = nlp.meta.nvar, nlp.nparam
  nnz_hess_k = Int(p * (p + 1) / 2) + (n - p) * p
  I = ((i,j) for i = 1:n, j = 1:p if j ≤ i)
  rows[1:nnz_hess_k] .= getindex.(I, 1)[:]
  cols[1:nnz_hess_k] .= getindex.(I, 2)[:]

  nini = hess_yu_obj_structure!(nlp, rows, cols, nfirst = nnz_hess_k)
  #if nini != nnzh_obj
  #  @warn "hess_(obj)_structure!: Size of vals and number of assignements didn't match"
  #end
  #################################################
  if nlp.meta.ncon > 0
    for term in nlp.op.terms
      if !(typeof(term) <: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm})
        continue
      end

      a = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
      ncells = num_cells(term.trian)
      cell_id_yu = Gridap.Arrays.IdentityVector(ncells)
    
      nini = struct_hess_coo_numeric!(rows, cols, a, cell_id_yu, nfirst = nini, cols_translate = nlp.nparam, rows_translate = nlp.nparam)
    end
  end

  if nlp.meta.nnzh != nini
    @warn "hess_structure!: Size of vals and number of assignements didn't match"
  end

  (rows, cols)
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

  nnzh_obj = get_nnzh(nlp.tnrj, nlp.Y, nlp.X, nlp.nparam, nlp.meta.nvar)
  hess_coord!(nlp, x, @view vals[1:nnzh_obj] )
  #vals[1:nnzh_obj] .= hess_coord(nlp, x) #TO BE REMOVED IF THAT WORK

#############################################################################
  κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
  yu     = FEFunction(nlp.Y, xyu)

  cell_yu    = Gridap.FESpaces.get_cell_values(yu)
  cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu)) #Is it?

  nini = nnzh_obj
  for term in nlp.op.terms
    if !(typeof(term) <: Union{Gridap.FESpaces.NonlinearFETermWithAutodiff, Gridap.FESpaces.NonlinearFETerm})
      continue
    end

    λf      = FEFunction(nlp.Xpde, λ)
    cell_λf = Gridap.FESpaces.get_cell_values(λf)
    lfh     = CellField(nlp.Xpde, cell_λf)
    _lfh    = Gridap.FESpaces.restrict(lfh, term.trian) #This is where the term play a first role.

    function _cell_res_yu(cell)
      xfh  = CellField(nlp.Y, cell)
      _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
      
      if length(κ) > 0
        _res = integrate(term.res(κ, _xfh,_lfh), term.trian, term.quad)
      else
        _res = integrate(term.res(_xfh,_lfh), term.trian, term.quad)
      end
      lag = _res
      return lag
    end

    #ncells     = num_cells(term.trian)
    #cell_id_yu = Gridap.Arrays.IdentityVector(ncells)

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_res_yu, cell_yu, cell_id_yu)
    #Assemble the matrix in the "good" space
    assem      = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
    #(I, J, V)  = assemble_hess(assem, cell_r_yu, cell_id_yu)
    nini = vals_hess_coo_numeric!(vals, assem, cell_r_yu, cell_id_yu, nfirst = nini)
    #vals[nini+1:nini+length(V)] .= V
    #nini += length(V)

    #= What about extra parameters????
    if nlp.nparam > 0
      (I1, J1, V1) = _compute_hess_k_coo(nlp, nlp.tnrj, κ, xyu)
      if obj_weight == one(eltype(x))
        return (vcat(I1, I2 .+ nlp.nparam),
                vcat(J1, J2 .+ nlp.nparam),
                vcat(V1, V2))
      else
        return (vcat(I1, I2 .+ nlp.nparam),
                vcat(J1, J2 .+ nlp.nparam),
                vcat(obj_weight * V1, obj_weight * V2))
      end
    end
    =#
  end
#############################################################################
  return vals
end

function cons!(nlp :: GridapPDENLPModel, x :: AbstractVector, c :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.ncon c
  increment!(nlp, :neval_cons)
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
  v      = Gridap.FESpaces.get_cell_basis(nlp.Xpde) #Tanj: is it really Xcon ?

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
###################################################################################
    κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
    @warn "Extra cons call"
    ck = @closure k -> cons(nlp, vcat(k, xyu))
    jac_k = ForwardDiff.jacobian(ck, κ)
###################################################################################
    return hcat(jac_k, pde_jac)
  end

  return pde_jac
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

"""
Adaptation of 
`function allocate_matrix(a::SparseMatrixAssembler,matdata) end`
in Gridap.FESpaces.
"""
function _jac_structure!(op :: Gridap.FESpaces.FEOperatorFromTerms, nlp :: GridapPDENLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})

  nini = jac_k_structure!(nlp, rows, cols)
  nini = allocate_coo_jac!(op, nlp.Y, nlp.Xpde, nlp.Ypde, nlp.Ycon, rows, cols, nfirst = nini, nparam = nlp.nparam)

  return rows, cols
end

function jac_k_structure!(nlp :: GridapPDENLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
    
  p = nlp.nparam
  n = nlp.meta.ncon
  nnz_jac_k = p*n
  I = ((i,j) for i = 1:n, j = 1:p)
  rows[1:nnz_jac_k] .= getindex.(I, 1)[:]
  cols[1:nnz_jac_k] .= getindex.(I, 2)[:]
    
  return nnz_jac_k
end

#=
function _jac_structure!(op :: Gridap.FESpaces.FEOperatorFromTerms, nlp :: GridapPDENLPModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})

 m, n = nlp.meta.ncon, nlp.meta.nvar
 I = ((i,j) for i = 1:m, j = 1:n)
 rows .= getindex.(I, 1)[:]
 cols .= getindex.(I, 2)[:]

 return rows, cols
end
=#

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
  
  nnz_jac_k = nlp.nparam > 0 ? nlp.meta.ncon * nlp.nparam : 0
  if nlp.nparam > 0
    κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
    ck = @closure k -> cons(nlp, vcat(k, xyu))
    jac_k = ForwardDiff.jacobian(ck, κ)
    vals[1:nnz_jac_k] .= jac_k[:]
  end
  
  #vals[1:nnz_hess_k] .= _compute_jac_k_vals(nlp, nlp.tnrj, κ, xyu)
  #Jx = jac(nlp, x)

  nini = _from_terms_to_jacobian_vals!(nlp.op, x, nlp.Y, nlp.Xpde, nlp.Ypde, nlp.Ycon, vals, nfirst = nnz_jac_k)
  
  if nlp.meta.nnzj != nini
    @warn "hess_coord!: Size of vals and number of assignements didn't match"
  end
  return vals
end

#=
function _jac_coord!(op :: Gridap.FESpaces.FEOperatorFromTerms, nlp :: GridapPDENLPModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nlp.meta.nvar x
  @lencheck nlp.meta.nnzj vals
  increment!(nlp, :neval_jac)
  Jx = jac(nlp, x)
  vals .= Jx[:]
  return vals
end
=#

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

  function ℓ(x, λ)

    κ, xyu = x[1 : nlp.nparam], x[nlp.nparam + 1 : nlp.meta.nvar]
    yu  = FEFunction(nlp.Y, xyu)
    int = _obj_integral(nlp.tnrj, κ, yu)
    
    c = similar(x, nlp.meta.ncon)
    _from_terms_to_residual!(nlp.op, x, nlp, c)

    return obj_weight * sum(int) + dot(c, λ)
  end
############# Tanj: test this #################################
 #agrad(t) = ForwardDiff.gradient(x->(obj_weight * obj(nlp, x) + dot(cons(nlp, x), λ)), x + t*v)
  agrad(t) = ForwardDiff.gradient(x->ℓ(x, λ) , x + t*v)
###############################################################
  Hv .= ForwardDiff.derivative(t -> agrad(t), 0.)

  return Hv
end

include("hess_difficulty.jl")

include("additional_functions.jl")
