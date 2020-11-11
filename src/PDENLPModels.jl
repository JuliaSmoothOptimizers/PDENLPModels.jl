module PDENLPModels

#This package contains a list of NLPModels for PDE optimization.
using ForwardDiff, LinearAlgebra, SparseArrays, FastClosures

#JSO packages
using NLPModels, LinearOperators

#PDE modeling
using Gridap

using NLPModels: increment!, decrement!, @lencheck
import NLPModels: obj, grad, grad!, hess, cons, cons!, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!, hprod, hprod!, hess_op, hess_op!

include("GridapPDENLPModel.jl")

export GridapPDENLPModel, obj, grad, grad!, hess, cons, cons!, jac, jprod, jprod!, jtprod, jtprod!, jac_op, jac_op!, hprod, hprod!, hess_op, hess_op!
export hess_old

end #end of module
