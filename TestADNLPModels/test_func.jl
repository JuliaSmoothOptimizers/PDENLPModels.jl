###############################################################################
#
# Tangi, November 30th 2020
#
# Functions to test and benchmark automatic differentiation
# ForwardDiff/ADNLPModel, SparseDiffTools, FiniteDiff
#
# Two issues:
#  - problem in defining the function for the jacobian, see jacobian_issue.jl
#  - FiniteDiff doesn't handle the **colorvec** yet.
#
###############################################################################

using LinearAlgebra, NLPModels, SparseArrays

using SparseDiffTools, SparsityDetection, FiniteDiff, ForwardDiff

using BenchmarkTools, Test

#global definitions are required for the @btime
function hv_test(nlp :: AbstractNLPModel)
    
    global f = x -> obj(nlp, x)
    global v1 = rand(nlp.meta.nvar)
    
    _1 = hprod(nlp, nlp.meta.x0, v1)
    _2 = num_hesvec(f, nlp.meta.x0, v1)
    _3 = autonum_hesvec(f, nlp.meta.x0, v1)
    _4 = numauto_hesvec(f, nlp.meta.x0, v1)

    @show norm(_1 - _2), norm(_1 - _3), norm(_1 - _4)

    @btime hprod(nlp, nlp.meta.x0, v1)
    @btime num_hesvec(f, nlp.meta.x0, v1)
    @btime autonum_hesvec(f, nlp.meta.x0, v1)
    @btime numauto_hesvec(f, nlp.meta.x0, v1)

    global _temp = similar(_1)
    
    @btime hprod!(nlp, nlp.meta.x0, v1, _temp)
    @btime num_hesvec!(_temp, f, nlp.meta.x0, v1)
    @btime autonum_hesvec!(_temp, f, nlp.meta.x0, v1)
    @btime numauto_hesvec!(_temp, f, nlp.meta.x0, v1)

    global nlp_op  = hess_op(nlp, nlp.meta.x0)
    global hop_true  = HesVec(f, nlp.meta.x0; autodiff=true)
    global hop_false = HesVec(f, nlp.meta.x0; autodiff=false)

    _5 = hop_true * v1
    _6 = hop_false * v1
    _7 = nlp_op * v1
    @show norm(_5 - _6), norm(_5 - _7)

    @btime hop_true * v1
    @btime hop_false * v1
    @btime nlp_op * v1
    
    return true
end

function jv_test(nlp :: AbstractNLPModel)
    
    global c = x -> cons(nlp, x)
    global v1 = rand(nlp.meta.nvar)
    
    _1 = jprod(nlp, nlp.meta.x0, v1)
    _2 = num_jacvec(c, nlp.meta.x0, v1)
    _3 = auto_jacvec(c, nlp.meta.x0, v1)

    @show norm(_1 - _2), norm(_1 - _3)
    
    @btime jprod(nlp, nlp.meta.x0, v1)
    @btime num_jacvec(c, nlp.meta.x0, v1)
    @btime auto_jacvec(c, nlp.meta.x0, v1)
    
    #global _temp = similar(_1)

    #@btime jprod!(nlp, nlp.meta.x0, v1, _temp)
    #@btime num_jacvec!(_temp, c, nlp.meta.x0, v1) #??
    #@btime auto_jacvec!(_temp, c, nlp.meta.x0, v1)

    global nlp_op  = jac_op(nlp, nlp.meta.x0)
    global hop_true  = JacVec(c, nlp.meta.x0; autodiff=true)
    global hop_false = JacVec(c, nlp.meta.x0; autodiff=false)

    _5 = hop_true * v1
    _6 = hop_false * v1
    _7 = nlp_op * v1
    @show norm(_1 - _5), norm(_1 - _6), norm(_1 - _7)

    @btime hop_true * v1
    @btime hop_false * v1
    @btime nlp_op * v1
    
    return true
end

#
# This one doesn't work, see jacobian_issue.jl to see why
#
function jac_test(nlp :: AbstractNLPModel)
 _Jx = NLPModels.jac(nlp, nlp.meta.x0)
 c(dx, x) = begin
     cx = cons(nlp, x)
     for i=1:nlp.meta.ncon
        dx[i] = cx[i] 
     end
     nothing
 end
 @warn "Doesn't work"
 jj(x :: AbstractVector, c::Function)= begin
 input = nlp.meta.x0
 output = similar(nlp.meta.y0)
 sparsity_pattern = jacobian_sparsity(c, output, input)
 T = eltype(nlp.meta.x0)
 jac = T.(sparse(sparsity_pattern))
 colors = matrix_colors(jac)
 forwarddiff_color_jacobian!(jac, c, nlp.meta.x0, colorvec = colors)
 return jac
 end
 jac =jj(nlp.meta.x0, c)
 
 @show norm(_Jx - jac)
 return true
end
