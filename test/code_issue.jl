#Prove that hess_coo is wrong...
global hess_is_zero = true
for k in 1:5
 I,J,Vi = Main.PDENLPModels.hess_coo(nlp, rand(nlp.meta.nvar), rand(nlp.meta.ncon))
 global hess_is_zero &= (Vi == zeros(length(Vi)))
end
@test hess_is_zero

x = get_free_values(zero(U))
λ = ones(511)

xf    =  FEFunction(U, x)
λf    = FEFunction(U, λ)

v     = Gridap.FESpaces.get_cell_basis(V)

cell_xf     = Gridap.FESpaces.get_cell_values(xf)
cell_λf     = Gridap.FESpaces.get_cell_values(λf)
ncells      = length(cell_xf)
cell_id_xf  = Gridap.Arrays.IdentityVector(ncells)

w, r, cc = [], [], []

term = op.terms[1] #let's try for one term to start.
_v   = restrict(v,  term.trian)

function _cell_res_xf(cell)
     xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    _res = integrate(res_pde(_xfh,_v), term.trian, term.quad)
    lag  = Array{Any,1}(nothing, ncells)#or Array{Any,1}(undef,ncells)
    for j in 1:ncells
        lag[j] = dot(_res[j], cell_λf[j])
    end
    return lag
end


#Compute the hessian with AD
cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_res_xf, cell_xf, cell_id_xf)
Gridap.FESpaces._push_matrix_contribution!(w,r,cc,cell_r_yu,cell_id_xf)

assem      = Gridap.FESpaces.SparseMatrixAssembler(U, V)
A = zeros(num_free_dofs(U),num_free_dofs(U))
Gridap.FESpaces.assemble_matrix!(A, assem, (w,r,cc))
