using Gridap, Printf

n = 4
domain    = (0,1,0,1)
partition = (n,n)
model     = CartesianDiscreteModel(domain,partition)

V0 = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model)

V = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

  g(x) = 1.0
U = TrialFESpace(V,g)

trian = Triangulation(model)
quad  = CellQuadrature(trian, 2)

function res(y,v)
 y*y*v - v*g
end
t_Ω  = FETerm(res,trian,quad)
op   = FEOperator(U,V,t_Ω)
