#=
#Prove that hess_coo is wrong...
global hess_is_zero = true
for k in 1:5
 I,J,Vi = Main.PDENLPModels.hess_coo(nlp, rand(nlp.meta.nvar), rand(nlp.meta.ncon))
 global hess_is_zero &= (Vi == zeros(length(Vi)))
end
@test hess_is_zero
=#
using Gridap

n = 3
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

V = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

  g(x) = 1.0
U = TrialFESpace(V,g)

trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

function res(y,v)
 y*y*v - v*g
end
t_Ω = FETerm(res,trian,quad)
op = FEOperator(U,V,t_Ω)

function func_lag(y, λ,v)
 (y*y*v - v*g) * λ
end

ncon = Gridap.FESpaces.num_free_dofs(U)

x = 0.5 * ones(ncon)
λ = ones(ncon)

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
    lag  = [_res[i]' * cell_λf[i] for i=1:ncells]
    #Array{Any,1}(nothing, ncells)#or Array{Any,1}(undef,ncells)
    #for j in 1:ncells
    #    lag[j] = dot(_res[j], cell_λf[j])
    #end
    return lag
end

function _cell_lag_res_xf(cell)
     xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    λ = -ones(ncon)
    λf    = FEFunction(V, λ)
    cell_λf     = Gridap.FESpaces.get_cell_values(λf)
    lfh  = CellField(U, cell_λf)
    _lfh = Gridap.FESpaces.restrict(lfh, term.trian)
    _res = integrate(func_lag(_xfh,_lfh,_v), term.trian, term.quad)
    return _res
end
#= 
How do I get the jacobian ?

function _cell_res_xf(xfh) #(cell)
     #xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    _res = integrate(res(_xfh,_v), term.trian, term.quad)
    #lag  = Array{Any,1}(nothing, ncells)#or Array{Any,1}(undef,ncells)
    #for j in 1:ncells
    #    lag[j] = dot(_res[j], cell_λf[j])
    #end
    return _res #lag
end

cellids  = Gridap.FESpaces.get_cell_id(term)

cellvals_y = Gridap.FESpaces.autodiff_cell_jacobian_from_residual(_cell_res_xf, xf, cellids)
=#

_res = _cell_res_xf(cell_xf)

#=
#Compute the hessian with AD
cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_res_xf, cell_xf, cell_id_xf)
Gridap.FESpaces._push_matrix_contribution!(w,r,c,cell_r_yu,cell_id_xf)

assem      = Gridap.FESpaces.SparseMatrixAssembler(U, V)
A = zeros(num_free_dofs(V),num_free_dofs(U))
Gridap.FESpaces.assemble_matrix!(A, assem, (w,r,c))
=#
#=
function _cell_res_xf(cell)
     xfh  = CellField(U, cell)
    _xfh = Gridap.FESpaces.restrict(xfh, term.trian)
    _res = integrate(res(_xfh,_v), term.trian, term.quad)
    #lag  = Array{Any,1}(nothing, ncells)#or Array{Any,1}(undef,ncells)
    #for j in 1:ncells
    #    lag[j] = dot(_res[j], cell_λf[j])
    #end
    return sum([_res[i]' * cell_λf[i] for i=1:ncells])
end
cell_r_yu  = Gridap.Arrays.autodiff_array_gradient(_cell_res_xf, cell_xf, cell_id_xf)


cell_r_yu  = Gridap.Arrays.autodiff_array_gradient(_cell_res_xf, cell_xf, cell_id_xf)
x2 = rand(ncon)
xf2    = FEFunction(U, x2)
cell_xf2     = Gridap.FESpaces.get_cell_values(xf2)
_cell_res_xf(cell_xf2)
=#

#=
 cellids  = Gridap.FESpaces.get_cell_id(term)
=#

#=
    cell_yu    = Gridap.FESpaces.get_cell_values(yu)
    cell_id_yu = Gridap.Arrays.IdentityVector(length(cell_yu))

    function _cell_obj_yu(cell)
         yuh = CellField(Y, cell)
        _obj_cell_integral(tnrj, κ, yuh)
  _yuh = Gridap.FESpaces.restrict(yuh, term.trian)

  integrate(term.f(_yuh), term.trian, term.quad)
    end

    #Compute the hessian with AD
    cell_r_yu  = Gridap.Arrays.autodiff_array_hessian(_cell_obj_yu, cell_yu, cell_id_yu)
    #Assemble the matrix in the "good" space
    assem      = Gridap.FESpaces.SparseMatrixAssembler(Y, X)
    (I, J, V) = assemble_hess(assem, cell_r_yu, cell_id_yu)
=#
