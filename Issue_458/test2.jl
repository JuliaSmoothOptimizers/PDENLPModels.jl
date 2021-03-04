#=
#Prove that hess_coo is wrong...
global hess_is_zero = true
for k in 1:5
 I,J,Vi = Main.PDENLPModels.hess_coo(nlp, rand(nlp.meta.nvar), rand(nlp.meta.ncon))
 global hess_is_zero &= (Vi == zeros(length(Vi)))
end
@test hess_is_zero
=#
include("data.jl")

function ℓ(y) #This is meant to be a Lagrangian
   y * y - g
end

ncon = Gridap.FESpaces.num_free_dofs(U)
x    = ones(ncon)
x    = rand(ncon)
λ    = ones(ncon) #ones(ncon)

xf    = FEFunction(U, x)
λf    = FEFunction(V, λ)
v     = Gridap.FESpaces.get_cell_basis(V)

cell_xf     = Gridap.FESpaces.get_cell_values(xf)
cell_λf     = Gridap.FESpaces.get_cell_values(λf)
ncells      = length(cell_xf)
cell_id_xf  = Gridap.Arrays.IdentityVector(ncells)

#We usually iterate over all op.terms
term = op.terms[1] #let's try for one term to start.
_v   = restrict(v,  term.trian)

function _cell_res_xf(cell)
    #reshape Lagrangian and solution:
     xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    lfh  = CellField(U, cell_λf)
    _lfh = Gridap.FESpaces.restrict(lfh, term.trian)
    
    #apply the function and integrate:
    #_res = integrate(res(_xfh,_v), term.trian, term.quad)
    #Option 1:
    _res = integrate(res(_xfh,_lfh), term.trian, term.quad) #Is it that we just apply the Lagrange multiplier instead of v ????
    lag = _res
    return lag
end

#Option 2: introduce a Lagrangian function:
function _cell_ℓ_xf(cell)
    #reshape Lagrangian and solution:
     xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    lfh  = CellField(U, cell_λf)
    _lfh = Gridap.FESpaces.restrict(lfh, term.trian)
    
    #apply the function and integrate:
    #Option 2: introduce a Lagrangian function:
    _res = integrate(ℓ(_xfh) * _lfh, term.trian, term.quad)
    lag = _res
    return lag
end

w, r, c = [], [], []

#Compute the hessian with AD
cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_res_xf, cell_xf, cell_id_xf)
Gridap.FESpaces._push_matrix_contribution!(w,r,c,cell_r_yu,cell_id_xf)

assem = Gridap.FESpaces.SparseMatrixAssembler(U, V)
A     = zeros(num_free_dofs(V),num_free_dofs(U))
Gridap.FESpaces.assemble_matrix!(A, assem, (w,r,c))

@show A #the hessian

#= Shows the gradient

cell_r_yu  = Gridap.Arrays.autodiff_array_gradient(_cell_res_xf, cell_xf, cell_id_xf)
vecdata_yu = [[cell_r_yu], [cell_id_xf]]
assem  = Gridap.FESpaces.SparseMatrixAssembler(U, V)
@show "Gradient", Gridap.FESpaces.assemble_vector(assem, vecdata_yu)
@show "Cells gradient", cell_r_yu

=#

w, r, c = [], [], []

#Compute the hessian with AD
cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_res_xf, cell_xf, cell_id_xf)
Gridap.FESpaces._push_matrix_contribution!(w,r,c,cell_r_yu,cell_id_xf)

assem = Gridap.FESpaces.SparseMatrixAssembler(U, V)
Aℓ     = zeros(num_free_dofs(V),num_free_dofs(U))
Gridap.FESpaces.assemble_matrix!(Aℓ, assem, (w,r,c))

@show Aℓ #the hessian

if norm(A - Aℓ) ≤ 1e-15 @printf "Yee-ha!" end

if false
  x2 = rand(ncon)
  xf2    = FEFunction(U, x2)
  cell_xf2     = Gridap.FESpaces.get_cell_values(xf2)
  xfh2  = CellField(U, cell_xf2)
  _xfh2 = Gridap.FESpaces.restrict(xfh2, term.trian)
  h = 1e-8
  x2p = x2 .+ h
  xf2p = FEFunction(U, x2p)
  cell_xf2p = Gridap.FESpaces.get_cell_values(xf2p)
  _top = (_cell_res_xf(cell_xf2p) - _cell_res_xf(cell_xf2)) ./ h

  @show "Solution:", _top
  @show "The ForwardDiff:"
  @show Gridap.Arrays.autodiff_array_gradient(_cell_res_xf, cell_xf2, cell_id_xf)
end

#=
using ForwardDiff
#include("Kernels.jl")
include("Autodiff2.jl")

@show "The FiniteDiff:"
cell_r_yu  = autodiff_array_gradient2(_cell_res_xf, cell_xf, cell_id_xf)
=#
