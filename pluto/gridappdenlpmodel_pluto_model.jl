using Gridap

domain = (0,1,0,1)
n = 2^5
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

###############################################################################
#Gridap resolution:
#This corresponds to a Poisson equation with Dirichlet and Neumann conditions
#described here: https://gridap.github.io/Tutorials/stable/pages/t001_poisson/
w(x) = 10.0
a(u,v) = ∇(v)⊙∇(u)
b_Ω(v) = v*w
t_Ω = AffineFETerm(a,b_Ω,trian,quad)

op_edp = AffineFEOperator(Ug,V0,t_Ω)
ls = LUSolver()
solver = LinearFESolver(ls)
uh = solve(solver,op_edp)

#writevtk(trian,"results",cellfields=["uh"=>uh])
###############################################################################

Yedp = Ug
Xedp = V0
Xcon = TestFESpace(
        reffe=:Lagrangian, order=1, valuetype=Float64,
        conformity=:H1, model=model)
Ycon = TrialFESpace(Xcon)

Y = MultiFieldFESpace([Yedp, Ycon])

ybis(x) =  -1#(x[1]-1)^2+(x[2]-1)^2
function f(yu) #:: Union{Gridap.FESpaces.SingleFieldFEFunction,Gridap.MultiField.MultiFieldFEFunction}
    y, u = yu
    1000 * 0.5 * (ybis - y) * (ybis - y) + 0.5 * u * u
end

function res_Ω(yu, v)
 y, u = yu
 v

 ∇(v)⊙∇(y) - v*u
end
w(x) = 10.0
b_Ω(v) = v*w
t_Ω = AffineFETerm(res_Ω, b_Ω, trian, quad)
op = AffineFEOperator(Y, V0, t_Ω)

using Main.PDENLPModels

nvar = Gridap.FESpaces.num_free_dofs(Y)
xin  = zeros(nvar)
nlp = GridapPDENLPModel(xin, f, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)

using FastClosures, Krylov, LinearAlgebra, LinearOperators
#This is a quadratic-linear optimizaion problem
xr = rand(nlp.meta.nvar)
A = Symmetric(hess(nlp, rand(nlp.meta.nvar)),:L)
b = grad(nlp, zeros(nlp.meta.nvar))
c = obj(nlp, zeros(nlp.meta.nvar))
f_test = 0.5*xr'*A*xr + b'*xr + c
fx = obj(nlp, xr)

function my_quick_algorithm_for_quad_lin_problems(nlp; x0 :: AbstractVector{T} = nlp.meta.x0) where T

 #First step, we form the system
 # ( G  A' ) (x) = (-g)
 # ( A  0  ) (λ) = ( b)
 #G = hess(nlp, x0)
 #A = jac(nlp,  x0)
 #g = grad(nlp, 0.)
 #b = cons(nlp, 0.)

 rhs = vcat(- grad(nlp, zeros(nlp.meta.nvar)), cons(nlp, zeros(nlp.meta.nvar)))

 #We now create a LinearOperator
 jacop = jac_op(nlp, x0)
 Jv  = Array{T,1}(undef, nlp.meta.ncon)
 Jtv = Array{T,1}(undef, nlp.meta.nvar)
 prod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jacop'*v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon],
                           jacop*v[1:nlp.meta.nvar])
 ctprod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jacop'*v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon],
                           jacop*v[1:nlp.meta.nvar])
 #prod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jtprod(nlp, x0, v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon]),
#                           jprod(nlp, x0, v[1:nlp.meta.nvar]))
 #ctprod = @closure v ->  vcat(hprod(nlp, x0, v[1:nlp.meta.nvar]) + jtprod(nlp, x0, v[nlp.meta.nvar+1:nlp.meta.nvar+nlp.meta.ncon]),
#                           jprod(nlp, x0, v[1:nlp.meta.nvar]))
#PreallocatedLinearOperator{T}(nrow::Int, ncol::Int, symmetric::Bool, hermitian::Bool, prod, tprod, ctprod)
 sad_op = PreallocatedLinearOperator{T}(nlp.meta.ncon+nlp.meta.nvar, nlp.meta.ncon+nlp.meta.nvar, true, true, prod, ctprod, ctprod)

 #(x, stats) = symmlq(A, rhs)
 (x, stats) = minres(sad_op, rhs)

 return x, stats
end

@time x, stats = my_quick_algorithm_for_quad_lin_problems(nlp)

sol = x[1:nlp.meta.nvar]
@show obj(nlp, x[1:nlp.meta.nvar])
yu = FEFunction(nlp.Y, sol)
y, u = yu

#writevtk(trian,"results-u",cellfields=["uh"=>u])
#writevtk(trian,"results-y",cellfields=["yh"=>y])

using NLPModelsIpopt
@time stats = ipopt(nlp)

nothing
