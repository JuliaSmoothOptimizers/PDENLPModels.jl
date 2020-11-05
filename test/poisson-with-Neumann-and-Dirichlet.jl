using Gridap
using BenchmarkTools, LinearAlgebra, NLPModels, Main.PDENLPModels, SparseArrays, Test

model = DiscreteModelFromFile("test/models/model.json")

#writevtk(model,"model")

V0 = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="sides")

  g(x) = 2.0
Ug = TrialFESpace(V0,g)

trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

neumanntags = ["circle", "triangle", "square"]
btrian = BoundaryTriangulation(model,neumanntags)
bquad = CellQuadrature(btrian,degree)

###############################################################################
#Gridap resolution:
#This corresponds to a Poisson equation with Dirichlet and Neumann conditions
#described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/
w(x) = 1.0
a(u,v) = ∇(v)⊙∇(u)
b_Ω(v) = v*w
t_Ω = AffineFETerm(a,b_Ω,trian,quad)

h(x) = 3.0
b_Γ(v) = v*h
t_Γ = FESource(b_Γ,btrian,bquad)

op_edp = AffineFEOperator(Ug,V0,t_Ω,t_Γ)
ls = LUSolver()
solver = LinearFESolver(ls)
uh = solve(solver,op_edp)

###############################################################################
# Create the NLP:

ybis(x) =  x[1]^2+x[2]^2
f(y,u) = 0.5 * (ybis - y) * (ybis - y) + 0.5 * u * u
function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
    y, u = yu
    f(y,u)
end

function res_Ω(yu, v)
 y, u = yu
 v

 ∇(v)⊙∇(y) - v*u
end
topt_Ω = FETerm(res_Ω, trian, quad)
function res_Γ(yu, v)
 y, u = yu
 v

 -v*h
end
topt_Γ = FETerm(res_Γ, btrian, bquad)#FESource(res_Γ, btrian, bquad)

Yedp = Ug
Xedp = V0
Xcon = TestFESpace(
        reffe=:Lagrangian, order=1, valuetype=Float64,
        conformity=:H1, model=model)
Ycon = TrialFESpace(Xcon)

Y = MultiFieldFESpace([Yedp, Ycon])
op = FEOperator(Y, V0, topt_Ω, topt_Γ) #or #op = FEOperator(Ug, V0, topt_Ω, topt_Γ)

xin = zeros(Gridap.FESpaces.num_free_dofs(Y));
@time nlp = GridapPDENLPModel(xin, zeros(0), f, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)

######
# 2nd test: we create a GridapPDENLPModel with affine operator
taff_Ω = AffineFETerm(res_Ω, v->0*v, trian, quad)
op_affine = AffineFEOperator(Y,V0,taff_Ω, t_Γ)

@test size(get_matrix(op_affine)) == (num_free_dofs(Xedp), num_free_dofs(Y))
@test length(get_vector(op_affine)) == num_free_dofs(Xedp)

nlp_affine = GridapPDENLPModel(xin, zeros(0), f, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op_affine)
################################################################################
# Check solution

sol_gridap = vcat(get_free_values(uh), 1.0*ones(nlp.nvar_control))

#Check the objective function:
@time _fx = obj(nlp, sol_gridap)
@time _gx = grad(nlp, sol_gridap)
#@test gradient_check(nlp) == Dict{Int64,Float64}() #works but slow
@time _Hx = hess(nlp, sol_gridap)
#The problem is quadratic so the hessian is constant:
@time _Hx2 = hess(nlp, rand(nlp.meta.nvar))
@test norm(_Hx - _Hx2) <= eps(Float64)
#H_errs_fg = hessian_check_from_grad(nlp) #needs hess_structure!
#@test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()

#We now test hprod for the objective hessian
@time _Hxv  = hprod(nlp, sol_gridap, ones(nlp.meta.nvar), obj_weight = 1.0)
@time _Hxv2 = hprod(nlp, rand(nlp.meta.nvar), ones(nlp.meta.nvar), obj_weight = 1.0)
@test norm(_Hxv - _Hxv2) <= eps(Float64) #since the hessian is constant
@test norm(_Hxv - Symmetric(_Hx,:L) * ones(nlp.meta.nvar), Inf) <= 5e-5#eps(Float64)

#We test the constraints (with the source term here):
@time _cx  = cons(nlp, sol_gridap)
op2 = FEOperator(Y, V0, topt_Ω, topt_Γ)
@time _cx2 = Gridap.FESpaces.residual(op2, FEFunction(Gridap.FESpaces.get_trial(op2), sol_gridap))
@test norm(_cx - _cx2) <= eps(Float64)
#Recall that cGx is also equal to "get_matrix(op_edp) * get_free_values(uh) - get_vector(op_edp)" here
@time cGx = Gridap.FESpaces.residual(op_edp, FEFunction(Gridap.FESpaces.get_trial(op_edp), get_free_values(uh)))
@test norm(_cx - cGx ,Inf) <= sqrt(eps(Float64))
@test norm(_cx, Inf) <= sqrt(eps(Float64))

@time _cxaff = cons(nlp_affine, sol_gridap)
xrand = rand(nlp.meta.nvar)
@time _affres = Gridap.FESpaces.residual(op_affine.op, xrand);
@time _cxraff = cons(nlp_affine, xrand)
@test norm(_cxraff - _affres , Inf) <= eps(Float64)
@test norm(_cxraff - cons(nlp, xrand), Inf) <= 1e-15


@time _Jx = PDENLPModels.jac(nlp, sol_gridap)
@time _Jxaff = PDENLPModels.jac(nlp_affine, sol_gridap)
@test norm(_Jx - get_matrix(nlp_affine.op)) <= eps(Float64)
@test norm(_Jxaff - get_matrix(nlp_affine.op)) <= eps(Float64)
#Test linear LinearOperators
@time jop = jac_op(nlp, sol_gridap)
v = rand(nlp.meta.nvar)
@test norm(jop * v - _Jx * v) <= eps(Float64)
_w = rand(nlp.meta.ncon)
@test norm(jop' * _w - _Jx' * _w) <= eps(Float64)
#@test jacobian_check(nlp) == Dict{Tuple{Int64,Int64},Float64}()#works but very slow

l = zeros(nlp.nvar_edp)
@time _Hxlv = hprod(nlp_affine, sol_gridap, l, v, obj_weight = 1.0)
@time _Hx   = hess(nlp_affine, sol_gridap)
@test norm(_Hxlv - Symmetric(_Hx,:L) * v) <= eps(Float64)


#We still miss the hprod! from the FEOperatorFromTerms
# :(

#function hess with multipliers not defined.
#H_errs = hessian_check(nlp) #slow
#@test H_errs[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
#H_errs_fg = hessian_check_from_grad(nlp)
#@test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()

nothing
