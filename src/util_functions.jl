"""
`_split_FEFunction(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})`

Split the vector x into two FEFunction corresponding to the solution `y` and the control `u`.
Returns nothing for the control `u` if Ycon == nothing.

Do not verify the compatible length.
"""
function _split_FEFunction(x::AbstractVector, Ypde::FESpace, Ycon::FESpace)
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nvar_con = Gridap.FESpaces.num_free_dofs(Ycon)

  yh = FEFunction(Ypde, x[1:nvar_pde])
  uh = FEFunction(Ycon, x[(1 + nvar_pde):(nvar_pde + nvar_con)])

  return yh, uh
end

# We should avoid this function, as it is not type stable output
function _split_FEFunction(x::AbstractVector, Ypde::FESpace, ::VoidFESpace)
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  yh = FEFunction(Ypde, x[1:nvar_pde])
  return yh, nothing
end

function _split_FEFunction(x, Ypde::FESpace, Ycon::FESpace)
  ny = typeof(Ypde) <: MultiFieldFESpace ? length(Ypde.spaces) : 1
  nu = typeof(Ycon) <: MultiFieldFESpace ? length(Ycon.spaces) : 1
  if typeof(Ypde) <: MultiFieldFESpace
    y = collect(first(x, ny))
  else
    y = first(x)
  end
  if typeof(Ycon) <: MultiFieldFESpace
    u = collect(first(Iterators.drop(x, ny), nu))
  else
    u = first(Iterators.drop(x, ny))
  end
  return y, u
end

# We should avoid this function, as it is not type stable output
function _split_FEFunction(x, Ypde::FESpace, ::VoidFESpace)
  if Ypde <: MultiFieldFESpace
    ny = length(Ypde.spaces)
    y = collect(first(x, ny))
  else
    y = x
  end
  return y, nothing
end

"""
`_split_vector(:: AbstractVector,  :: FESpace, :: Union{FESpace, Nothing})`

Split the vector x into three vectors: y, u, k.
Returns nothing for the control `u` if Ycon == nothing.

Do not verify the compatible length.
"""
function _split_vector(x::AbstractVector, Ypde::FESpace, Ycon::FESpace)
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nvar_con = Gridap.FESpaces.num_free_dofs(Ycon)
  #nparam   = length(x) - (nvar_pde + nvar_con)

  y = x[1:nvar_pde]
  u = x[(1 + nvar_pde):(nvar_pde + nvar_con)]
  k = x[(nvar_pde + nvar_con + 1):length(x)]

  return y, u, k
end

function _split_vector(x::AbstractVector, Ypde::FESpace, Ycon::VoidFESpace)
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  #nparam   = length(x) - nvar_pde

  y = x[1:nvar_pde]
  k = x[(nvar_pde + 1):length(x)]

  return y, [], k
end

"""
    `_split_vectors(x, Ypde, Ycon)`

Take a vector x and returns a splitting in terms of `y`, `u` and `θ`.
"""
function _split_vectors(x::AbstractVector, Ypde::FESpace, Ycon::FESpace)
  nvar_pde = Gridap.FESpaces.num_free_dofs(Ypde)
  nvar_con = Gridap.FESpaces.num_free_dofs(Ycon)

  nparam = length(x) - nvar_pde - nvar_con
  θ = x[(nvar_pde + nvar_con + 1):end]

  leng(S::MultiFieldFESpace, i) = sum([Gridap.FESpaces.num_free_dofs(S.spaces[j]) for j = 1:i])
  sum_old(S::MultiFieldFESpace, i) =
    if i == 1
      0
    else
      sum([Gridap.FESpaces.num_free_dofs(S[j]) for j = 1:(i - 1)])
    end

  y = if typeof(Ypde) <: MultiFieldFESpace
    (x[(sum_old(Ypde, i) + 1):leng(Ypde, i)] for i = 1:length(Ypde.spaces))
  else
    x[1:nvar_pde]
  end
  if nvar_con == 0 && nparam == 0
    return y
  elseif nvar_con == 0 && nparam > 0
    return y, θ
  end
  u = if typeof(Ycon) <: MultiFieldFESpace
    (x[(sum_old(Ycon, i) + 1 + nvar_pde):leng(Ycon, i)] for i = 1:length(Ycon.spaces))
  else
    x[(1 + nvar_pde):(nvar_pde + nvar_con)]
  end
  if nparam == 0
    return y, u
  end

  return y, u, θ
end

export dt

# for the weak formulation of dy/dt
conv(u, ∇u) = (∇u ⋅ one(∇u)) ⊙ u # (∇u')⋅u
dt(u, v) = conv ∘ (v, ∇(u))
