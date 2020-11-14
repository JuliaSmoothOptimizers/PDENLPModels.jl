using Gridap, Test

n = 20
domain = (0,1)
partition = n
model = CartesianDiscreteModel(domain,partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[2])
add_tag_from_tags!(labels,"diri0",[1])

V = TestFESpace(
  reffe=:Lagrangian, conformity=:H1, valuetype=Float64,
  model=model, labels=labels, order=1, dirichlet_tags=["diri0","diri1"])

uD0 = VectorValue(0)
uD1 = VectorValue(-1)
U = TrialFESpace(V,[uD0,uD1])

###############################################################################
#Solve using Gridap
@law conv(y,∇y) = (∇y ⋅one(∇y))⊙y

nu = 0.08
h(x) = 2*(nu + x[1]^3)
function res_pde(y,v)
  u(x) = 0.5
  -nu*(∇(v)⊙∇(y)) + v⊙conv(y,∇(y)) - v * u - v * h
end

trian = Triangulation(model)
degree = 1
quad = CellQuadrature(trian,degree)
t_Ω = FETerm(res_pde,trian,quad)
op_pde = FEOperator(U,V,t_Ω)

#Check resolution for z given.
using LineSearches: BackTracking
nls = NLSolver(
  show_trace=true, method=:newton, linesearch=BackTracking())
solver = FESolver(nls)

uh = solve(solver,op_pde)
sol_gridap = get_free_values(uh)
#sol_gridap = vcat(get_free_values(uh), 0.5*ones(513))

#writevtk(trian,"data/results_nl",cellfields=["uh"=>uh])
###############################################################################

Xedp = V
Yedp = U
Xcon = TestFESpace(
        reffe=:Lagrangian, order=1, valuetype=Float64,
        conformity=:H1, model=model)
Ycon = TrialFESpace(Xcon)
Y = MultiFieldFESpace([Yedp, Ycon])

#Now we move to the optimization:
yd(x) = -x[1]^2
α = 1/1e-2
#objective function:
f(y, u) = 0.5 * (yd - y) * (yd - y) + 0.5 * α * (u - 0.5) * (u - 0.5)
function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
    y, u = yu
    f(y,u)
end

function res(yu, v) #u is the solution of the PDE and z the control
 y, u = yu
 v

 -nu*(∇(v)⊙∇(y)) + v⊙conv(y,∇(y)) - v * u - v * h
end
t_Ω = FETerm(res,trian,quad)
op = FEOperator(Y, V, t_Ω)

@test Gridap.FESpaces.num_free_dofs(U) < Gridap.FESpaces.num_free_dofs(Ycon)
#################################################################################

using NLPModels, Krylov, Main.PDENLPModels
xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
@time nlp = GridapPDENLPModel(xin, f, Yedp, Ycon, Xedp, Xcon, trian, quad, op = op)

include("../SQPalgorithm/SQP-factorization-free.jl")
using Main.SQPFactFree

#Great initial guess
x0 = vcat(sol_gridap, 0.5*ones(Gridap.FESpaces.num_free_dofs(Ycon)))

@time x = sqp_solver(nlp, x0 = x0, atol = 1e-3, max_iter = 10, itmax = 1000)

@test obj(nlp, x) < obj(nlp, x0)

using NLPModelsIpopt
@time stats = ipopt(nlp)
