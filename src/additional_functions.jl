#ℓ(x) = obj_weight * nlp.f(x)
#Hv .= ForwardDiff.derivative(t -> ForwardDiff.gradient(ℓ, x + t * v), 0)
"""
Compute hessian-vector product of the objective function.

Note: this is not efficient at all.
Test on n=14115
@btime hprod(nlp, sol_gridap, v)
  42.683 s (274375613 allocations: 29.89 GiB)
while computing the hessian and then the product yields
@btime _Hx = hess(nlp, sol_gridap);
  766.036 ms (724829 allocations: 121.84 MiB)
@btime hprod(nlp, sol_gridap, v)
  42.683 s (274375613 allocations: 29.89 GiB)

"""
function hprod_autodiff!(nlp :: GridapPDENLPModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector; obj_weight :: Real = one(eltype(x)))
  @lencheck nlp.meta.nvar x v Hv
  increment!(nlp, :neval_hprod)

  if obj_weight == zero(eltype(x))
      Hv .= zero(similar(x))
      return Hv
  end

  assem = Gridap.FESpaces.SparseMatrixAssembler(nlp.Y, nlp.X)
  #We prepare computation of x + t * v
  #At t=0.
  yu    = FEFunction(nlp.Y, x)
  vf    = FEFunction(nlp.Y, v)

  cell_yu   = Gridap.FESpaces.get_cell_values(yu)
  cell_vf   = Gridap.FESpaces.get_cell_values(vf)
  ncells    = length(cell_yu)
  cell_id   = Gridap.Arrays.IdentityVector(ncells)

  function _cell_obj_t(cell)
       th = CellField(nlp.Y, cell)
      _th = Gridap.FESpaces.restrict(th, nlp.trian)
      integrate(nlp.f(_th), nlp.trian, nlp.quad) #function needs to return array of size 1.
  end

  #Compute the gradient with AD
  function _cell_grad_t(t)
      ct     = t * ones(length(cell_vf[1]))
      _cell  = Array{typeof(ct .* cell_vf[1])}(undef, ncells)
      for i=1:ncells
          _cell[i] = cell_yu[i] + ct .* cell_vf[i]
      end
      Gridap.Arrays.autodiff_array_gradient(_cell_obj_t, _cell, cell_id)
  end

  #Compute the derivative w.r.t. to t of _cell_grad_t
  #This might be slow as it cannot be decomposed (easily) cell by cell
  cell_r_yu = ForwardDiff.derivative(_cell_grad_t, 0.)

  #Put the result in the format expected by Gridap.FESpaces.assemble_matrix
  vecdata_yu = [[cell_r_yu], [cell_id]]
  #Assemble the gradient in the "good" space
  Hv .= Gridap.FESpaces.assemble_vector(assem, vecdata_yu)

  return Hv
end

"""
`_get_y_and_u(:: GridapPDENLPModel, :: AbstractVector{T}) `

Returns y and u in matrix format where
y ∈ n_edp_fields x nvar_per_field
u ∈ n_control_fields x nvar_per_field
It is useful when evaluating the function constraint (and jacobian) functions.
"""
function _get_y_and_u(nlp :: GridapPDENLPModel, x :: AbstractVector{T}) where T

    y = Array{T,2}(undef, nlp.n_edp_fields, nlp.nvar_per_field)
    for i=1:nlp.n_edp_fields
        y[i,:] = [x[k] for k in i:nlp.n_edp_fields:nlp.nvar_edp]
    end

    u = Array{T,2}(undef, nlp.n_control_fields, nlp.nvar_per_field)
    for i=1:nlp.n_control_fields
        u[i,:] = [x[k] for k in nlp.nvar_edp+i:nlp.n_control_fields:nlp.meta.nvar-nlp.nparam]
    end

    return y, u
end

function _get_y_and_u_i(nlp :: GridapPDENLPModel, x :: AbstractVector{T}, j :: Int) where T

    y = Array{T,1}(undef, nlp.n_edp_fields)
    for i=1:nlp.n_edp_fields
        y[i] = x[(j-1)*nlp.n_edp_fields + i]
    end

    u = Array{T,1}(undef, nlp.n_control_fields)
    for i=1:nlp.n_control_fields
        u[i] = x[nlp.nvar_edp+(j-1)*nlp.n_control_fields + i]
    end

    return y, u
end

function _get_y_and_u_i(nlp :: GridapPDENLPModel, x :: AbstractVector{T}, v :: AbstractVector{T}, j :: Int) where T

    _v = Array{T,1}(undef, nlp.n_edp_fields + nlp.n_control_fields)

    y = Array{T,1}(undef, nlp.n_edp_fields)
    for i=1:nlp.n_edp_fields
        y[i] = x[(j-1)*nlp.n_edp_fields + i]
       _v[i] = v[(j-1)*nlp.n_edp_fields + i]
    end

    u = Array{T,1}(undef, nlp.n_control_fields)
    for i=1:nlp.n_control_fields
        u[i] = x[nlp.nvar_edp + (j-1) * nlp.n_control_fields + i]
       _v[i] = v[nlp.n_edp_fields + nlp.nvar_edp + (j-1) * nlp.n_control_fields + i]
    end

    return y, u, _v
end
