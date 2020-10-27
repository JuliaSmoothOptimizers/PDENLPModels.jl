using Gridap
n = 3
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[6,])
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

D = 2
order = 2
V = TestFESpace(
  reffe=:Lagrangian, conformity=:H1, valuetype=VectorValue{D,Float64},
  model=model, labels=labels, order=order, dirichlet_tags=["diri0","diri1"])

Q = TestFESpace(
  reffe=:PLagrangian, conformity=:L2, valuetype=Float64,
  model=model, order=order-1, constraint=:zeromean)

uD0 = VectorValue(0,0)
uD1 = VectorValue(1,0)
U = TrialFESpace(V,[uD0,uD1])
P = TrialFESpace(Q)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

const Re = 10.0
@law conv(u,∇u) = Re*(∇u')⋅u
@law dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

function a(x,y)
  u, p = x
  v, q = y
  ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u)
end

c(u,v) = v⊙conv(u,∇(u))
dc(u,du,v) = v⊙dconv(du,∇(du),u,∇(u))

function res(x,y)
  u, p = x
  v, q = y
  a(x,y) + c(u,v)
end

function jac(x,dx,y)
  u, p = x
  v, q = y
  du, dp = dx
  a(dx,y)+ dc(u,du,v)
end

trian = Triangulation(model)
degree = (order-1)*2
quad = CellQuadrature(trian,degree)
t_Ω = FETerm(res,jac,trian,quad)
op = FEOperator(X,Y,t_Ω)

using LinearAlgebra, NLPModels, Main.PDENLPModels, SparseArrays, Test
ndofs = Gridap.FESpaces.num_free_dofs(X)
xin   = zeros(ndofs)
Ycon, Xcon = nothing, nothing
#In this example Y and X are switched
@time nlp = GridapPDENLPModel(xin, zeros(0), x->0.0, X, Ycon, Y, Xcon, trian, quad, op = op)

@time fx = obj(nlp, xin)
@test fx == 0.0 #just to check :)
@time cx  = cons(nlp, xin)
@time Gcx = Gridap.FESpaces.residual(op, FEFunction(Gridap.FESpaces.get_trial(op), xin))
@warn "Maybe compare time of cx and Gcx"
@test length(cx) == ndofs
@time Jx   = Main.PDENLPModels.jac(nlp, xin) #avoid confusion with the function jac from the problem
@time GJx = Gridap.FESpaces.jacobian(nlp.op, FEFunction(Gridap.FESpaces.get_trial(op), xin))
@warn "Compare AD jacobian and analytical one (when available)"
@test norm(GJx - Jx, Inf) <= sqrt(eps(Float64))
@test size(Jx) == (ndofs, ndofs)
@time Jxx  = jprod(nlp, xin, xin)
@test norm(Jx * xin - Jxx) <= sqrt(eps(Float64))
@test norm(GJx * xin - Jxx) <= sqrt(eps(Float64))
@time Jtxo = jtprod(nlp, xin, zeros(ndofs))
@test norm(Jtxo) <= sqrt(eps(Float64))
@time Jtxu = jtprod(nlp, xin, ones(ndofs))
@test norm(GJx' * ones(ndofs) - Jtxu, Inf) <= sqrt(eps(Float64))

#jac_op =
(rows, cols, vals) = findnz(Jx)
@time jacop = jac_op(nlp, xin)

#Gridap way of solving the equation:
using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)
@time uh, ph = solve(solver,op)
sol_gridap = vcat(get_free_values(uh), get_free_values(ph))

@warn "Also tries an homemade/JSOSolvers algorithm."

cGx = Gridap.FESpaces.residual(op, FEFunction(Gridap.FESpaces.get_trial(op), sol_gridap))
cx  = cons(nlp, sol_gridap)
nGcx = norm(cGx, Inf)
ncx  = norm(cx, Inf)
@test norm(cx - cGx) == 0.0
@test nGcx == ncx
JGsolx = Gridap.FESpaces.jacobian(op, FEFunction(Gridap.FESpaces.get_trial(op), sol_gridap))
Jsolx = Main.PDENLPModels.jac(nlp, sol_gridap)
@test norm(JGsolx - Jsolx,Inf) <= sqrt(eps(Float64))

#This is an example where the jacobian is not symmetric
@test norm(Jx' - Jx) > 1.
