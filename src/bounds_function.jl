export bounds_functions_to_vectors

"""
`(lvar, uvar) = bounds_functions_to_vectors(Y :: MultiFieldFESpace, Ycon :: Union{FESpace, Nothing},  Ypde :: FESpace, trian :: Triangulation, lyfunc :: Union{Function, AbstractVector}, uyfunc :: Union{Function, AbstractVector}, lufunc :: Union{Function, AbstractVector}, uufunc :: Union{Function, AbstractVector})`

Return the bounds `lvar` and `uvar`.
"""
function bounds_functions_to_vectors(
  Y::FESpace, #Y      :: MultiFieldFESpace, 
  Ycon::FESpace,
  Ypde::FESpace,
  trian::Triangulation,
  lyfunc::Function,
  uyfunc::Function,
  lufunc::Function,
  uufunc::Function,
)

  #Get the coordinates of the cells
  cell_xs = get_cell_coordinates(trian)
  #Create a function that given a cell returns the middle.
  midpoint(xs) = sum(xs) / length(xs)
  cell_xm = apply(midpoint, cell_xs) #this is a vector of size num_cells(trian)

  nfields_y = if typeof(Ypde) <: MultiFieldFESpace
    Gridap.MultiField.num_fields(Ypde) #Gridap.MultiField.num_fields(Y)
  else #  typeof(Ypde) <: FESpace
    1
  end
  nfields_u = if typeof(Ycon) <: MultiFieldFESpace
    Gridap.MultiField.num_fields(Ycon) #Gridap.MultiField.num_fields(Y)
  else #  typeof(Ycon) <: FESpace
    1
  end

  @assert nfields_y == length(lyfunc(cell_xm[1]))
  @assert nfields_y == length(uyfunc(cell_xm[1]))
  @assert nfields_u == length(lufunc(cell_xm[1]))
  @assert nfields_u == length(uufunc(cell_xm[1]))

  T = eltype(lyfunc(cell_xm[1])) #Hopefully, it is the same type for all functions

  #Allocate lvar and uvar
  ny, nu = Gridap.FESpaces.num_free_dofs(Ypde), Gridap.FESpaces.num_free_dofs(Ycon)
  lvar = Array{T, 1}(undef, ny + nu)
  uvar = Array{T, 1}(undef, ny + nu)

  nini = 0
  nini = _functions_to_vectors!(nini, nfields_y, trian, lyfunc, uyfunc, cell_xm, Ypde, lvar, uvar)
  nini = _functions_to_vectors!(nini, nfields_u, trian, lufunc, uufunc, cell_xm, Ycon, lvar, uvar)

  @assert nini == ny + nu

  return lvar, uvar
end

function bounds_functions_to_vectors(
  Y::FESpace, #Y      :: MultiFieldFESpace, 
  Ycon::FESpace,
  Ypde::FESpace,
  trian::Triangulation,
  ly::AbstractVector{T},
  uy::AbstractVector{T},
  lufunc::Function,
  uufunc::Function,
) where {T}

  #Allocate lvar and uvar
  ny, nu = Gridap.FESpaces.num_free_dofs(Ypde), Gridap.FESpaces.num_free_dofs(Ycon)
  lvar = Array{T, 1}(undef, ny + nu)
  uvar = Array{T, 1}(undef, ny + nu)

  #Get the coordinates of the cells
  cell_xs = get_cell_coordinates(trian)
  #Create a function that given a cell returns the middle.
  midpoint(xs) = sum(xs) / length(xs)
  cell_xm = apply(midpoint, cell_xs) #this is a vector of size num_cells(trian)

  nfields_u = if typeof(Ycon) <: MultiFieldFESpace
    Gridap.MultiField.num_fields(Ycon) #Gridap.MultiField.num_fields(Y)
  else #  typeof(Ycon) <: FESpace
    1
  end

  @lencheck nfields_u lufunc(cell_xm[1]) uufunc(cell_xm[1])

  nini = 0

  @lencheck length(ly) uy
  nini += length(ly)
  lvar[1:nini] .= ly
  uvar[1:nini] .= uy
  nini = _functions_to_vectors!(nini, nfields_u, trian, lufunc, uufunc, cell_xm, Ycon, lvar, uvar)

  @assert nini == ny + nu

  return lvar, uvar
end

function bounds_functions_to_vectors(
  Y::FESpace, #Y      :: MultiFieldFESpace, 
  Ycon::FESpace,
  Ypde::FESpace,
  trian::Triangulation,
  ly::AbstractVector{T},
  uy::AbstractVector{T},
  lu::AbstractVector{T},
  uu::AbstractVector{T},
) where {T}
  ny, nu = Gridap.FESpaces.num_free_dofs(Ypde), Gridap.FESpaces.num_free_dofs(Ycon)

  @lencheck ny ly uy
  @lencheck nu lu uu

  return vcat(ly, lu), vcat(uy, uu)
end

function bounds_functions_to_vectors(
  Y::FESpace, #Y      :: MultiFieldFESpace, 
  Ycon::VoidFESpace,
  Ypde::FESpace,
  trian::Triangulation,
  lyfunc::Function,
  uyfunc::Function,
  lufunc::Any,
  uufunc::Any,
)

  #Get the coordinates of the cells
  cell_xs = get_cell_coordinates(trian)
  #Create a function that given a cell returns the middle.
  midpoint(xs) = sum(xs) / length(xs)
  cell_xm = apply(midpoint, cell_xs) #this is a vector of size num_cells(trian)

  nfields_y = if typeof(Ypde) <: MultiFieldFESpace
    Gridap.MultiField.num_fields(Ypde) #Gridap.MultiField.num_fields(Y)
  else #  typeof(Ypde) <: FESpace
    1
  end

  @assert nfields_y == length(lyfunc(cell_xm[1]))
  @assert nfields_y == length(uyfunc(cell_xm[1]))

  T = eltype(lyfunc(cell_xm[1])) #Hopefully, it is the same type for all functions

  #Allocate lvar and uvar
  ny = Gridap.FESpaces.num_free_dofs(Ypde)
  nu = 0
  lvar = Array{T, 1}(undef, ny + nu)
  uvar = Array{T, 1}(undef, ny + nu)

  nini = 0
  nini = _functions_to_vectors!(nini, nfields_y, trian, lyfunc, uyfunc, cell_xm, Ypde, lvar, uvar)

  @assert nini == ny + nu

  return lvar, uvar
end

function bounds_functions_to_vectors(
  Y::FESpace, #Y      :: MultiFieldFESpace, 
  Ycon::VoidFESpace,
  Ypde::FESpace,
  trian::Triangulation,
  ly::AbstractVector,
  uy::AbstractVector,
  lu::AbstractVector,
  uu::AbstractVector,
)
  ny = Gridap.FESpaces.num_free_dofs(Ypde)

  @lencheck ny ly uy

  return ly, uy
end

"""
_functions_to_vectors!(nini :: Int, nfields :: Int, trian :: Triangulation, lfunc :: Function, ufunc :: Function, cell_xm :: Gridap.Arrays.AppliedArray, Y :: FESpace, lvar :: AbstractVector, uvar :: AbstractVector)

Iterate for `k = 1` to `nfields` and switch `lfunc[k]` and `ufunc[k]` to vectors, 
allocated in `lvar` and `uvar` in place starting from `nini + 1`.
It returns nini + the number of allocations.
"""
function _functions_to_vectors!(
  nini::Integer,
  nfields::Integer,
  trian::Triangulation,
  lfunc::Function,
  ufunc::Function,
  cell_xm, # What is this type ??? Former Gridap.Arrays.AppliedArray
  Y::FESpace,
  lvar::AbstractVector,
  uvar::AbstractVector,
)
  n = length(lvar)
  @lencheck n uvar

  for i = 1:nfields
    Yi = typeof(Y) <: MultiFieldFESpace ? Y.spaces[i] : Y

    cell_l = apply(x -> lfunc(x)[i], cell_xm) #this is a vector of size num_cells(trian)
    cell_u = apply(x -> ufunc(x)[i], cell_xm) #this is a vector of size num_cells(trian)

    lvaru = get_free_values(Gridap.FESpaces.interpolate(cell_l, Yi))
    uvaru = get_free_values(Gridap.FESpaces.interpolate(cell_u, Yi))
    new = length(lvaru)

    @assert new == length(uvaru)
    @assert new ≤ n

    lvar[(nini + 1):(nini + new)] .= lvaru
    uvar[(nini + 1):(nini + new)] .= uvaru

    nini += new
  end

  return nini
end
