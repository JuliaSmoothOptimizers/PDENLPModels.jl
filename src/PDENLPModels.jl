module PDENLPModels

#This package contains a list of NLPModels for PDE optimization.
using ForwardDiff, LinearAlgebra, SparseArrays, FastClosures

#JSO packages
using NLPModels, LinearOperators

#PDE modeling
using Gridap

using NLPModels: increment!, decrement!, @lencheck
import NLPModels: obj, grad, grad!, hess, cons, cons!, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!, hprod, hprod!, hess_op, hess_op!, jac_structure, jac_structure!, jac_coord, jac_coord!, hess_structure, hess_structure!, hess_coord, hess_coord!

include("GridapPDENLPModel.jl")

export GridapPDENLPModel, obj, grad, grad!, hess, cons, cons!, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!, hprod, hprod!, hess_op, hess_op!, jac_structure, jac_structure!, jac_coord, jac_coord!, hess_structure, hess_structure!, hess_coord, hess_coord!
export hess_old

end #end of module
