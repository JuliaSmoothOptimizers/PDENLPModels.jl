fcalls = 0
function f(dx,x) # in-place
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end

function g(x) # out-of-place
  global fcalls += 1
  dx = zero(x)
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  dx
end

using SparsityDetection, SparseArrays
input = rand(30)
output = similar(input)
sparsity_pattern = jacobian_sparsity(f,output,input)
jac = Float64.(sparse(sparsity_pattern))

using SparseDiffTools
colors = matrix_colors(jac)

using FiniteDiff
FiniteDiff.finite_difference_jacobian!(jac, f, rand(30), colorvec=colors)
@show fcalls # 5

#in-place
forwarddiff_color_jacobian!(jac, f, x, colorvec = colors)
#out-of-place
#forwarddiff_color_jacobian(g, x, colorvec = colors) #More kwargs to be checked

u = rand(30)
J = JacVec(f,u)
#HesVec(f,u::AbstractArray;autodiff=true)

using LinearAlgebra
v = rand(30)
res = similar(v)
mul!(res,J,v) # Does 1 f evaluation
#Additional operators for HesVec exists, including HesVecGrad which allows one to utilize a gradient function. 

#=
#Matrix-vector product as a function:
auto_jacvec!(du, f, x, v)
auto_jacvec(f, x, v)
num_jacvec(f,x,v)

#and for the hessian:
num_hesvec!(du,f,x,v)
num_hesvec(f,x,v)
autonum_hesvec!(du,f,x,v)
autonum_hesvec(f,x,v)
=#
