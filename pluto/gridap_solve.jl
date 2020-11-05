###############################################################################
#Gridap resolution:
#This corresponds to a Poisson equation with Dirichlet and Neumann conditions
#described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/
using Gridap

domain = (0,1,0,1)
n = 2^7
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

V0 = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

g(x) = 0.0
Ug = TrialFESpace(V0,g)

w(x) = 1.0
a(u,v) = ∇(v)⊙∇(u)
b_Ω(v) = v*w
t_Ω = AffineFETerm(a,b_Ω,trian,quad)

op_edp = AffineFEOperator(Ug,V0,t_Ω)
ls = LUSolver()
solver = LinearFESolver(ls)
uh = solve(solver,op_edp)

writevtk(trian,"results",cellfields=["uh"=>uh])
