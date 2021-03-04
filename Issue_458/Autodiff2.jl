#Addon by Tanj:
using FiniteDiff
import Gridap: Gridap.Arrays.IdentityVector

import Gridap:  Gridap.Arrays.kernel_cache,  Gridap.Arrays.apply_kernel!
#end
"""
"""
function autodiff_array_gradient2(a,i_to_x,j_to_i=IdentityVector(length(i_to_x)))

  i_to_xdual = apply(i_to_x) do x
    cfg = ForwardDiff.GradientConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
    xdual = cfg.duals
    @show xdual
    xdual
  end
  #j_to_f =  Gridap.Arrays.to_array_of_functions(a,i_to_xdual,j_to_i)
  j_to_f =  Gridap.Arrays.to_array_of_functions(a,i_to_x,j_to_i)
  j_to_x =  Gridap.Arrays.reindex(i_to_x,j_to_i)

  k = ForwardDiffGradientKernel2() # Gridap.Arrays.ForwardDiffGradientKernel()
  apply(k,j_to_f,j_to_x)

end

struct ForwardDiffGradientKernel2 <: Gridap.Arrays.Kernel end

function kernel_cache(k:: ForwardDiffGradientKernel2,f,x)
  r = copy(x)
  #cfg = ForwardDiff.GradientConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
  cfg = FiniteDiff.GradientCache(r, x, Val(:central), eltype(r), Val(true))
  (r, cfg)
end

@inline function apply_kernel!(cache,k:: ForwardDiffGradientKernel2,f,x)
  #r, cfg = cache
  r = copy(x)
  #@notimplementedif length(r) != length(x) #commented by Tanj
  #ForwardDiff.gradient!(r,f,x,cfg)
  #FiniteDiff.finite_difference_gradient!(r, f, x, cfg)
  FiniteDiff.finite_difference_gradient!(r, f, x)
  r
end

"""
"""
function autodiff_array_jacobian2(a,i_to_x,j_to_i=IdentityVector(length(i_to_x)))

  i_to_xdual = apply(i_to_x) do x
    cfg = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
    xdual = cfg.duals
    xdual
  end

  j_to_f =  Gridap.Arrays.to_array_of_functions(a,i_to_xdual,j_to_i)
  j_to_x =  Gridap.Arrays.reindex(i_to_x,j_to_i)

  k =  Gridap.Arrays.ForwardDiffJacobianKernel()
  apply(k,j_to_f,j_to_x)

end

#struct ForwardDiffJacobianKernel <: Kernel end

function kernel_cache(k:: Gridap.Arrays.ForwardDiffJacobianKernel,f,x)
  cfg = ForwardDiff.JacobianConfig(nothing, x, ForwardDiff.Chunk{length(x)}())
  n = length(x)
  j = zeros(eltype(x),n,n)
  (j, cfg)
end

@inline function apply_kernel!(cache,k:: Gridap.Arrays.ForwardDiffJacobianKernel,f,x)
  j, cfg = cache
  #@notimplementedif size(j,1) != length(x) #commented by Tanj
  #@notimplementedif size(j,2) != length(x) #commented by Tanj
  ForwardDiff.jacobian!(j,f,x,cfg)
  j
end

"""
"""
function autodiff_array_hessian2(a,i_to_x,j_to_i=IdentityVector(length(i_to_x)))
   agrad = i_to_y -> autodiff_array_gradient2(a,i_to_y,j_to_i)
   autodiff_array_jacobian2(agrad,i_to_x,j_to_i)
end

#=
function to_array_of_functions(a,x,ids=IdentityVector(length(x)))
  k = ArrayOfFunctionsKernel(a,x)
  j = IdentityVector(length(ids))
  apply(k,j)
end

=#
#=
struct ArrayOfFunctionsKernel{A,X} <: Kernel
  a::A
  x::X
end
=#
#=
function kernel_cache(k:: Gridap.Arrays.ArrayOfFunctionsKernel,j)
  xi = testitem(k.x)
  l = length(k.x)
  x = MutableFill(xi,l)
  ax = k.a(x)
  axc = array_cache(ax)
  (ax, x, axc)
end

@inline function apply_kernel!(cache,k:: Gridap.Arrays.ArrayOfFunctionsKernel,j)
  ax, x, axc = cache
  @inline function f(xj)
    x.value = xj
    axj = getindex!(axc,ax,j)
  end
  f
end


mutable struct MutableFill{T} <: AbstractVector{T}
  value::T
  length::Int
end

Base.size(a::MutableFill) = (a.length,)

@inline Base.getindex(a::MutableFill,i::Integer) = a.value
=#
