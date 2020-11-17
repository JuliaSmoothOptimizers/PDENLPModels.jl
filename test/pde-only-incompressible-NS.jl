using BenchmarkTools, Gridap, LinearAlgebra, NLPModels, Main.PDENLPModels, SparseArrays, Test

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

X = MultiFieldFESpace([V, Q])
Y = MultiFieldFESpace([U, P])

const Re = 10.0
@law conv(u,∇u) = Re*(∇u')⋅u
@law dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

function a(y,x)
  u, p = y
  v, q = x
  ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u)
end

c(u,v) = v⊙conv(u,∇(u))
dc(u,du,v) = v⊙dconv(du,∇(du),u,∇(u))

function res(y,x)
  u, p = y
  v, q = x
  a(y,x) + c(u,v)
end

function jac(y,dy,x)
  u, p = y
  v, q = x
  du, dp = dy
  a(dy,x)+ dc(u,du,v)
end

trian = Triangulation(model)
degree = (order-1)*2
quad = CellQuadrature(trian,degree)
t_Ω = FETerm(res,trian,quad)
op = FEOperator(Y,X,t_Ω)
t_with_jac_Ω = FETerm(res,jac,trian,quad)
op_with_jac = FEOperator(Y,X,t_with_jac_Ω)


ndofs = Gridap.FESpaces.num_free_dofs(Y)
xin   = zeros(ndofs)
Ycon, Xcon = nothing, nothing
@time nlp = GridapPDENLPModel(xin, x->0.0, trian, quad, Y, Ycon, X, Xcon, op)

@time fx = obj(nlp, xin)
@test fx == 0.0
@time gx = grad(nlp, xin)
@test gx == zeros(nlp.meta.nvar)
@test gradient_check(nlp) == Dict{Int64,Float64}()
@time _Hxv = hprod(nlp, rand(nlp.meta.nvar), ones(nlp.meta.nvar));
@test _Hxv == zeros(nlp.meta.nvar)

#We also compare cons and Gridap.FESpaces.residual using @btime:
@btime cons(nlp, xin)
@btime Gridap.FESpaces.residual(op, FEFunction(Gridap.FESpaces.get_trial(op), xin))
#expected result:
#1.377 ms (10895 allocations: 1.21 MiB) for  cons :)
#2.492 ms (19611 allocations: 2.16 MiB) for residual
cx  = cons(nlp, xin)
Gcx = Gridap.FESpaces.residual(op, FEFunction(Gridap.FESpaces.get_trial(op), xin))
@test norm(cx - Gcx, Inf) == 0.0
@test length(cx) == ndofs

#We also compare jac and Gridap.FESpaces.jacobian using @btime:
#Note: Avoid confusion with the function jac from the problem
@btime Main.PDENLPModels.jac(nlp, xin); #for now we use AD to compute jacobian
@btime Gridap.FESpaces.jacobian(op, FEFunction(Gridap.FESpaces.get_trial(op), xin));
@btime Gridap.FESpaces.jacobian(op_with_jac, FEFunction(Gridap.FESpaces.get_trial(op), xin));
#expected results:
#9.656 ms (28898 allocations: 12.78 MiB) for jac :)
#25.290 ms (71788 allocations: 31.69 MiB) for jacobian without analytical jacobian
#8.562 ms (56321 allocations: 6.61 MiB) for jacobian with analytical jacobian
Jx  = Main.PDENLPModels.jac(nlp, xin)
GJx = Gridap.FESpaces.jacobian(op, FEFunction(Gridap.FESpaces.get_trial(op), xin))
GJx_with_jac = Gridap.FESpaces.jacobian(op_with_jac, FEFunction(Gridap.FESpaces.get_trial(op), xin))
@test issparse(Jx)
@test size(Jx) == (ndofs, ndofs)
@test norm(GJx - Jx, Inf) <= eps(Float64)
@test norm(GJx_with_jac - Jx, Inf) <= eps(Float64)
@test jacobian_check(nlp) == Dict{Tuple{Int64,Int64},Float64}()

@time Jxx  = jprod(nlp, xin, xin)
@test norm(Jx * xin - Jxx, Inf) <= eps(Float64)
@test norm(GJx * xin - Jxx, Inf) <= eps(Float64)
@time Jtxo = jtprod(nlp, xin, zeros(ndofs))
@test norm(Jtxo, Inf) <= eps(Float64)
@time Jtxu = jtprod(nlp, xin, ones(ndofs))
@test norm(GJx' * ones(ndofs) - Jtxu, Inf) <= eps(Float64)

#jac_op = (rows, cols, vals) = findnz(Jx)
@time jacop = jac_op(nlp, xin)
@time jacoptxu = jacop.tprod(ones(ndofs))
@test norm(Jtxu - jacoptxu, Inf) <= eps(Float64)

#Gridap way of solving the equation:
using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)
@time uh, ph = solve(solver,op)
sol_gridap = vcat(get_free_values(uh), get_free_values(ph))

cGx = Gridap.FESpaces.residual(op, FEFunction(Gridap.FESpaces.get_trial(op), sol_gridap))
cx  = cons(nlp, sol_gridap)
@test norm(cx - cGx, Inf) <= eps(Float64)

JGsolx = Gridap.FESpaces.jacobian(op, FEFunction(Gridap.FESpaces.get_trial(op), sol_gridap))
Jsolx = Main.PDENLPModels.jac(nlp, sol_gridap)
@test norm(JGsolx - Jsolx,Inf) <= eps(Float64)

#This is an example where the jacobian is not symmetric
@test norm(Jx' - Jx) > 1.

@warn "hprod returns NaNs"
#_Hxou = hprod(nlp, xin, zeros(nlp.meta.ncon), ones(ndofs))
#@test norm(_Hxou, Inf) <= eps(Float64)
#_Hxrr = hprod(nlp, xin, rand(nlp.meta.ncon), rand(ndofs))
#@test norm(_Hxrr, Inf) <= eps(Float64)

#H_errs = hessian_check(nlp) #slow
#@test H_errs[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
#H_errs_fg = hessian_check_from_grad(nlp)
#@test H_errs_fg[0] == Dict{Int, Dict{Tuple{Int,Int}, Float64}}()
