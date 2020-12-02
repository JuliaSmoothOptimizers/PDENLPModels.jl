###############################################################################
#
# This test case consider the optimization of a parameter in a Poisson equation
# with Dirichlet boundary conditions.
#
# Aim:
# * Test mixed problem
# * with integral term in the objective function
# * |k| = 2
#
###############################################################################
using Gridap, Main.PDENLPModels, LinearAlgebra, SparseArrays, NLPModels

n = 5
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain,partition)

#We use a manufactured solution of the PDE:
sol(x) = sin(2*pi*x[1]) * x[2]

V0 = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model, dirichlet_tags="boundary")

Ug = TrialFESpace(V0, sol)

trian = Triangulation(model)
degree = 2
quad = CellQuadrature(trian,degree)

#We deduce the rhs of the Poisson equation with our manufactured solution:
f(x) = (2*pi^2) * sin(2*pi*x[1]) * x[2]

function res(k, y, v)
 k[1] * ∇(v)⊙∇(y) - v*f*k[2]
end
t_Ω = FETerm(res, trian, quad)
op = FEOperator(Ug, V0, t_Ω)

function fk(k, y)
 0.5 * (sol - y) * (sol - y) + 0.5 * (k[1] - 1.) * (k[1] - 1.) + 0.5 * (k[2] - 1.) * (k[2] - 1.)
end
Vp = TestFESpace(
  reffe=:Lagrangian, order=1, valuetype=Float64,
  conformity=:H1, model=model)
Large = MultiFieldFESpace(repeat([Vp], 2))
#nrj = MixedEnergyFETerm(fk, trian, quad, 2, Large) #length(k)=2
nrj = MixedEnergyFETerm(fk, trian, quad, 2)

nUg = num_free_dofs(Ug)
x0  = zeros(nUg + 2) #zeros(nUg + 2)
nlp = GridapPDENLPModel(x0, nrj, Ug, V0, op)

using Test
@test nlp.nparam == 2
@test nlp.tnrj.nparam == 2
x1 = vcat(1., 1., rand(nUg))
@test obj(nlp, x0) > 0.5
@test obj(nlp, x1) > 0.
@test grad(nlp, x0)[1:2] ≈ - ones(2) atol = 1e-14
@test grad(nlp, x1)[1:2] ≈ zeros(2)  atol = 1e-14

hess(nlp, x0)

hessian_test_functions(nlp)
