module PDENLPModels

#This package contains a list of NLPModels for PDE optimization.
using ForwardDiff, LinearAlgebra, SparseArrays, FastClosures

#JSO packages
using NLPModels

#PDE modeling
using Gridap

function Gridap.Arrays.testargs(k::Gridap.Arrays.PosNegReindex, i::Integer)
  # @check length(k.values_pos) !=0 || length(k.values_neg) != 0 "This map has empty domain"
  #if !(eltype(k.values_pos) == eltype(k.values_neg))
  #@show typeof(k.values_pos), typeof(k.values_neg)
  #@show one(i)
  #@show eltype(k.values_pos), eltype(k.values_neg)
  #end
  # @check eltype(k.values_pos) == eltype(k.values_neg) "This map is type-instable"
  length(k.values_pos) != 0 ? (one(i),) : (-one(i))
end

function Gridap.FESpaces.scatter_free_and_dirichlet_values(
  f::Gridap.FESpaces.UnconstrainedFESpace,
  free_values,
  dirichlet_values,
)
  #=
  @check eltype(free_values) == eltype(dirichlet_values) """\n
  The entries stored in free_values and dirichlet_values should be of the same type.

  This error shows up e.g. when trying to build a FEFunction from a vector of integers
  if the Dirichlet values of the underlying space are of type Float64, or when the
  given free values are Float64 and the Dirichlet values ComplexF64.
  """
  =#
  cell_dof_ids = get_cell_dof_ids(f)
  lazy_map(Broadcasting(Gridap.Arrays.PosNegReindex(free_values, dirichlet_values)), cell_dof_ids)
end

include("gridap_autodiff.jl")
include("gridap_utils.jl")

#Regroup the different types of FEFunction
const FEFunctionType =
  Union{Gridap.FESpaces.SingleFieldFEFunction, Gridap.MultiField.MultiFieldFEFunction}
const CellFieldType = Union{Gridap.MultiField.MultiFieldCellField, Gridap.CellData.GenericCellField}

struct VoidFESpace <: FESpace end

import Gridap.FESpaces.num_free_dofs

num_free_dofs(::VoidFESpace) = 0

struct VoidMultiFieldFESpace <: FESpace #MultiFieldFESpace
  spaces::Any

  function VoidMultiFieldFESpace()
    return new([])
  end
end

import Gridap.MultiField.num_fields

num_fields(::VoidMultiFieldFESpace) = 0

_fespace_to_multifieldfespace(Y::MultiFieldFESpace) = Y
_fespace_to_multifieldfespace(Y::VoidMultiFieldFESpace) = Y
_fespace_to_multifieldfespace(::VoidFESpace) = VoidMultiFieldFESpace()
_fespace_to_multifieldfespace(Y::FESpace) = MultiFieldFESpace([Y])

# function fill_matrix_coo_numeric!(I,J,V,a::GenericSparseMatrixAssembler,matdata,n=0) specialized from L.428 Gridap/FESpaces/SparseMatrixAssembler.jl
include("fill_matrix_hessian_functions.jl")

#Additional modeling structures for the objective function.
# include("hessian_functions.jl")
include("additional_obj_terms.jl")
#Set of practical functions: _split_FEFunction, _split_vector
include("util_functions.jl")
include("hessian_struct_nnzh_functions.jl")
#Additional functions for the jacobian:
include("jacobian_functions.jl")

export AbstractEnergyTerm, EnergyFETerm, MixedEnergyFETerm, NoFETerm

#Import NLPModels functions surcharged by the GridapPDENLPModel
using NLPModels: increment!, decrement!, @lencheck
import NLPModels:
  obj,
  grad,
  grad!,
  hess,
  cons,
  cons!,
  jac,
  jprod,
  jprod!,
  jtprod,
  jtprod!,
  jac_op,
  jac_op!,
  hprod,
  hprod!,
  hess_op,
  hess_op!,
  jac_structure,
  jac_structure!,
  jac_coord,
  jac_coord!,
  hess_structure,
  hess_structure!,
  hess_coord,
  hess_coord!

include("GridapPDENLPModel.jl")

export GridapPDENLPModel,
  obj,
  grad,
  grad!,
  hess,
  cons,
  cons!,
  jac,
  jprod,
  jprod!,
  jtprod,
  jtprod!,
  jac_op,
  jac_op!,
  hprod,
  hprod!,
  hess_op,
  hess_op!,
  jac_structure,
  jac_structure!,
  jac_coord,
  jac_coord!,
  hess_structure,
  hess_structure!,
  hess_coord,
  hess_coord!
export hess_old, hess_coo, hess_obj_structure, hess_obj_structure!

#meta functions
import NLPModels:
  has_bounds,
  bound_constrained,
  unconstrained,
  linearly_constrained,
  equality_constrained,
  inequality_constrained
export has_bounds,
  bound_constrained,
  unconstrained,
  linearly_constrained,
  equality_constrained,
  inequality_constrained

#counters functions
import NLPModels: reset!
export reset!

export split_vectors

"""
    `split_vectors(::GridapPDENLPModel, x)`

Take a vector x and returns a splitting in terms of `y`, `u` and `θ`.
"""
function split_vectors(nlp::GridapPDENLPModel, x::AbstractVector)
  return _split_vectors(x, nlp.pdemeta.Ypde, nlp.pdemeta.Ycon)
end

end #end of module
