module PDENLPModels

  #This package contains a list of NLPModels for PDE optimization.
  using ForwardDiff, LinearAlgebra, SparseArrays, FastClosures

  #JSO packages
  using NLPModels, LinearOperators

  #PDE modeling
  using Gridap

  #Regroup the different types of FEFunction
  const FEFunctionType = Union{Gridap.FESpaces.SingleFieldFEFunction,
                               Gridap.MultiField.MultiFieldFEFunction}
  const CellFieldType  = Union{Gridap.MultiField.MultiFieldCellField,
                               Gridap.CellData.GenericCellField}

  """
  Useful structure to avoid putting nothing
  """
  struct VoidFESpace <: FESpace end

  import Gridap.FESpaces.num_free_dofs

  num_free_dofs(:: VoidFESpace) = 0

  struct VoidMultiFieldFESpace <: FESpace #MultiFieldFESpace
    spaces
    
    function VoidMultiFieldFESpace() return new([]) end
  end

  import Gridap.MultiField.num_fields

  num_fields(:: VoidMultiFieldFESpace) = 0

  _fespace_to_multifieldfespace(Y :: MultiFieldFESpace) = Y
  _fespace_to_multifieldfespace(Y :: VoidMultiFieldFESpace) = Y
  _fespace_to_multifieldfespace(:: VoidFESpace) = VoidMultiFieldFESpace()
  _fespace_to_multifieldfespace(Y :: FESpace) = MultiFieldFESpace([Y])

  #Additional modeling structures for the objective function.
  include("hessian_functions.jl")
  include("additional_obj_terms.jl")
  #Set of practical functions: _split_FEFunction, _split_vector
  include("util_functions.jl")
  #Additional functions for the jacobian:
  include("jacobian_functions.jl")

  export AbstractEnergyTerm, EnergyFETerm, MixedEnergyFETerm, NoFETerm

  #Import NLPModels functions surcharged by the GridapPDENLPModel
  using NLPModels: increment!, decrement!, @lencheck
  import NLPModels: obj, grad, grad!, hess, cons, cons!, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!, hprod, hprod!, hess_op, hess_op!, jac_structure, jac_structure!, jac_coord, jac_coord!, hess_structure, hess_structure!, hess_coord, hess_coord!

  include("GridapPDENLPModel.jl")

  export GridapPDENLPModel, obj, grad, grad!, hess, cons, cons!, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!, hprod, hprod!, hess_op, hess_op!, jac_structure, jac_structure!, jac_coord, jac_coord!, hess_structure, hess_structure!, hess_coord, hess_coord!
  export hess_old, hess_coo, hess_obj_structure, hess_obj_structure!

  #meta functions
  import NLPModels: has_bounds, bound_constrained, unconstrained, linearly_constrained, equality_constrained, inequality_constrained
  export has_bounds, bound_constrained, unconstrained, linearly_constrained, equality_constrained, inequality_constrained

  #counters functions
  import NLPModels: reset!
  export reset!

  using Test
  include("test_functions/hessian_test_functions.jl")
  include("test_functions/hessian_lag_test_functions.jl")
  include("test_functions/jacobian_test_functions.jl")
  export hessian_test_functions, hessian_lagrangian_test_functions, jacobian_test_functions

  #Set of functions for optimal control
  include("optimal_control_functions.jl")
end #end of module
