include("data.jl")

ncon = Gridap.FESpaces.num_free_dofs(U)
x    = ones(ncon)
λ    = ones(ncon)

xf    = FEFunction(U, x)
λf    = FEFunction(V, λ)
v     = Gridap.FESpaces.get_cell_basis(V)

cell_xf     = Gridap.FESpaces.get_cell_values(xf)
cell_λf     = Gridap.FESpaces.get_cell_values(λf)
ncells      = length(cell_xf)
cell_id_xf  = Gridap.Arrays.IdentityVector(ncells)

w, r, c = [], [], []

term = op.terms[1] #let's try for one term to start.
_v   = restrict(v,  term.trian)

function _cell_res_xf(cell)
     xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    _res = integrate(res(_xfh,_v), term.trian, term.quad)
    lag  = [dot(_res[j], cell_λf[j]) for j in 1:ncells ]
    return lag
end

#Compute the hessian with AD
cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_res_xf, cell_xf, cell_id_xf)
Gridap.FESpaces._push_matrix_contribution!(w,r,c,cell_r_yu,cell_id_xf)

assem = Gridap.FESpaces.SparseMatrixAssembler(U, V)
A     = zeros(num_free_dofs(V),num_free_dofs(U))
Gridap.FESpaces.assemble_matrix!(A, assem, (w,r,c))

#=
An explanation of why this is not working due to ForwardDiff:
function f(y) #This is meant to be a Lagrangian
   y * y - g
end

#Two examples just for the gradient of an objective function:
#The example working:
function _cell_res_xf(cell)
     xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    cell_λf1 = ones(length(cell_λf[1]))
    lag = integrate(f(_xfh), term.trian, term.quad)
    return lag
end
#and the one that doesn't
function _cell_res_xf(cell)
     xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    cell_λf1 = ones(length(cell_λf[1]))
    _res = integrate(f(_xfh), term.trian, term.quad)
    lag = [_res[j] for j=1:ncells]
    return lag
end
=#
