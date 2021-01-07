################################################################################
#Inspired by
#https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/master/src/qn_model.jl
# but we keep the hessian of the objective function.
#
# We aim here an QuasiNewtonPDEModel using a quasi-Newton technique to model
# the Lagrangian Hessian.
# The `QuasiNewtonModel` defined in `NLPModels` considers only unconstrained
# problems.
#
# \TODO: this piece of code is not finished yet.
#
# How to model this? Using a list of LBFGSOperator (one for each constraint) ?
# Have an option for automatic push! whenever the jacobian is called.
# How to give inverseLBFGSOperator automatically?
# How should we test this?
#
################################################################################

export QuasiNewtonPDEModel, LBFGSPDEModel, LSR1PDEModel

abstract type QuasiNewtonPDEModel <: AbstractNLPModel end

mutable struct LBFGSPDEModel <: QuasiNewtonPDEModel
  meta :: NLPModelMeta
  model :: AbstractNLPModel
  op :: LBFGSOperator
end

mutable struct LSR1PDEModel <: QuasiNewtonPDEModel
  meta :: NLPModelMeta
  model :: AbstractNLPModel
  op :: LSR1Operator
end

"Construct a `LBFGSPDEModel` from another type of model."
function LBFGSPDEModel(nlp :: AbstractNLPModel; kwargs...)
  op = LBFGSOperator(nlp.meta.nvar; kwargs...)
  return LBFGSPDEModel(nlp.meta, nlp, op)
end

"Construct a `LSR1PDEModel` from another type of nlp."
function LSR1PDEModel(nlp :: AbstractNLPModel; kwargs...)
  op = LSR1Operator(nlp.meta.nvar; kwargs...)
  return LSR1PDEModel(nlp.meta, nlp, op)
end

show_header(io :: IO, nlp :: QuasiNewtonPDEModel) = println(io, "$(typeof(nlp)) - A QuasiNewtonPDEModel")

function show(io :: IO, nlp :: QuasiNewtonPDEModel)
  show_header(io, nlp)
  show(io, nlp.meta)
  show(io, nlp.model.counters)
end

@default_counters QuasiNewtonPDEModel model

function reset_data!(nlp :: QuasiNewtonPDEModel)
  reset!(nlp.op)
  return nlp
end

# the following methods are not affected by the Hessian approximation
for meth in (:obj, :grad, :cons, :jac_coord, :jac)
  @eval $meth(nlp :: QuasiNewtonPDEModel, x :: AbstractVector) = $meth(nlp.model, x)
end
for meth in (:grad!, :cons!, :jprod, :jtprod, :objgrad, :objgrad!)
  @eval $meth(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, y :: AbstractVector) = $meth(nlp.model, x, y)
end
for meth in (:jprod!, :jtprod!)
  @eval $meth(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, y :: AbstractVector, z :: AbstractVector) = $meth(nlp.model, x, y, z)
end
jac_structure!(nlp :: QuasiNewtonPDEModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer}) = jac_structure!(nlp.model, rows, cols)
jac_coord!(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, vals :: AbstractVector) =
    jac_coord!(nlp.model, x, vals)
#The hessian functions that remain valid
hess(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, obj_weight :: Real = one(eltype(x))) = hess(nlp.model, x, obj_weight = obj_weight)
hess_op(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, obj_weight :: Real = one(eltype(x))) = hess_op(nlp.model, x, obj_weight = obj_weight)
hess_op!(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, Hv :: AbstractVector, obj_weight :: Real = one(eltype(x))) = hess_op!(nlp.model, x, Hv, obj_weight = obj_weight)
hprod(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, v :: AbstractVector, obj_weight :: Real = one(eltype(x))) = hprod(nlp.model, x, v, obj_weight = obj_weight)
hprod!(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, v :: AbstractVector, Hv :: AbstractVector, obj_weight :: Real = one(eltype(x))) = hprod!(nlp.model, x, v, Hv, obj_weight = obj_weight)
#works well with hess_obj_structure, but maybe not hess_structure
hess_coord(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, obj_weight::Real = one(eltype(x))) = hess_coord(nlp.model, x, obj_weight = obj_weight)
hess_coord!(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, vals :: AbstractVector, obj_weight::Real = one(eltype(x))) = hess_coord!(nlp.model, x, vals, obj_weight = obj_weight)
hess_obj_structure(nlp :: QuasiNewtonPDEModel) = hess_obj_structure(nlp.model)
hess_obj_structure!(nlp :: QuasiNewtonPDEModel, rows :: AbstractVector, cols :: AbstractVector) = hess_obj_structure(nlp.model, rows, cols)

# the following methods are affected by the Hessian approximation
#hess_op(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, y :: AbstractVector; kwargs...) = nlp.op
hess_op(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, y :: AbstractVector; obj_weight::Real = one(eltype(x))) = nlp.op #hess_op(nlp, x, obj_weight=obj_weight) + nlp.op
#hprod(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, v :: AbstractVector; kwargs...) = nlp.op * v
hprod(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, y :: AbstractVector, v :: AbstractVector; obj_weight::Real = one(eltype(x))) = nlp.op * v
#function hprod!(nlp :: QuasiNewtonPDEModel, x :: AbstractVector,
#                v :: AbstractVector, Hv :: AbstractVector; kwargs...)
#  Hv[1:nlp.meta.nvar] .= nlp.op * v
#  return Hv
#end
function hprod!(nlp :: QuasiNewtonPDEModel, x :: AbstractVector, y :: AbstractVector,
                v :: AbstractVector, Hv :: AbstractVector; kwargs...)
  Hv[1:nlp.meta.nvar] .= nlp.op * v
  return Hv
end

function push!(nlp :: QuasiNewtonPDEModel, args...)
	push!(nlp.op, args...)
	return nlp
end

# not implemented: hess_structure, hess_coord (with constraints), hess (with constraints)