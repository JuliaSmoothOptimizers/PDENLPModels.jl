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

 ∇(v)⊙∇(u) - v*z
end
topt_Ω = FETerm(res_Ω, trian, quad)
function res_Γ(yu, v) #u is the solution of the PDE and z the control
 u, z = yu
 v

 v*h
end
topt_Γ = FESource(res_Γ, btrian, bquad)
op = FEOperator(U, V, topt_Ω,topt_Γ)

Yedp = MultiFieldFESpace([Ug])
Xedp = V0
Xcon = TestFESpace(
        reffe=:Lagrangian, order=1, valuetype=Float64,
        conformity=:H1, model=model)
Ycon = MultiFieldFESpace([TrialFESpace(Xcon)])

Y = MultiFieldFESpace(vcat(Yedp.spaces, Ycon.spaces))
xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
@time nlp = GridapPDENLPModel(xin, zeros(0), f, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)

sol_gridap = vcat(get_free_values(uh), 1.0*ones(nlp.nvar_control))

@time _Hx = hess(nlp, sol_gridap)
@test issymmetric(_Hx)

l = zeros(nlp.nvar_edp)
#Error here:
#@time _Hxlv = hprod(nlp, sol_gridap, l, ones(nlp.meta.nvar), obj_weight = 1.0)
