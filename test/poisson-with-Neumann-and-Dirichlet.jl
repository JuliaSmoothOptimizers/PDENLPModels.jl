using Gridap

model = DiscreteModelFromFile("test/models/model.json")

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

function res_Ω(yu, v) #u is the solution of the PDE and z the control
 u, z = yu
 v

 ∇(v)⊙∇(u) + v*z #∇(v)⊙∇(u) - v*z
end
topt_Ω = FETerm(res_Ω, trian, quad)
function res_Γ(yu, v) #u is the solution of the PDE and z the control
 u, z = yu
 v

 v*h
end
topt_Γ = FETerm(res_Γ, btrian, bquad)#FESource(res_Γ, btrian, bquad)
op = FEOperator(Ug, V0, topt_Ω,topt_Γ)

Yedp = Ug
Xedp = V0
Xcon = TestFESpace(
        reffe=:Lagrangian, order=1, valuetype=Float64,
        conformity=:H1, model=model)
Ycon = TrialFESpace(Xcon)

Y = MultiFieldFESpace([Yedp, Ycon])
xin = zeros(Gridap.FESpaces.num_free_dofs(Y))

using BenchmarkTools, LinearAlgebra, NLPModels, Main.PDENLPModels, SparseArrays, Test
@time nlp = GridapPDENLPModel(xin, zeros(0), f, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)

sol_gridap = vcat(get_free_values(uh), 1.0*ones(nlp.nvar_control))

@time _fx = obj(nlp, sol_gridap)
@time _gx = grad(nlp, sol_gridap)
#@test gradient_check(nlp) == Dict{Int64,Float64}() #works but slow
@time _Hx = hess(nlp, sol_gridap)
@test issymmetric(_Hx)
@time _Hxv = hprod(nlp, sol_gridap, ones(nlp.meta.nvar), obj_weight = 1.0)
@test norm(_Hxv - _Hx * ones(nlp.meta.nvar), Inf) <= 5e-5#eps(Float64)

@warn "Something is wrong here:"
@time _cx = cons(nlp, sol_gridap)
#Recall that cGx is also equal to "get_matrix(op_edp) * get_free_values(uh) - get_vector(op_edp)" here
@time cGx = Gridap.FESpaces.residual(op_edp, FEFunction(Gridap.FESpaces.get_trial(op_edp), get_free_values(uh)))
@test norm(_cx,Inf) <= 1e-2
@time _Jx = jac(nlp, sol_gridap)
@test jacobian_check(nlp) == Dict{Tuple{Int64,Int64},Float64}()

l = zeros(nlp.nvar_edp)
#Error here:
#@time _Hxlv = hprod(nlp, sol_gridap, l, ones(nlp.meta.nvar), obj_weight = 1.0)

#function hess with multipliers not defined.
#H_errs = hessian_check(nlp) #slow
#@test H_errs[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
#H_errs_fg = hessian_check_from_grad(nlp)
#@test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()

nothing
