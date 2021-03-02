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

check_nlp_dimensions(nlp, exclude_hess=true)

using Test
@test nlp.nparam == 2
@test nlp.tnrj.nparam == 2
x1 = vcat(1., 1., rand(nUg))
@test obj(nlp, x0) > 0.5
@test obj(nlp, x1) > 0.
@test grad(nlp, x0)[1:2] ≈ - ones(2) atol = 1e-14
@test grad(nlp, x1)[1:2] ≈ zeros(2)  atol = 1e-14

_Hx = hess(nlp, x0)

@test Matrix(_Hx[1:2,1:2]) ≈ diagm(0=>ones(2)) atol=1e-14

#We compare the NLP with a related one.
function fk2(y)
 0.5 * (sol - y) * (sol - y)
end
nrj2 = EnergyFETerm(fk2, trian, quad)
nlp2 = GridapPDENLPModel(zeros(16), nrj2, Ug, V0, op)

κ, xyu = nlp2.meta.x0[1 : nlp2.nparam], nlp2.meta.x0[nlp2.nparam + 1 : nlp2.meta.nvar]
yu     = FEFunction(nlp.Y, xyu)
(_I,_J,_V) = Main.PDENLPModels._compute_hess_coo(nlp.tnrj, zeros(2), yu, nlp.Y, nlp.X)
_HH = hess(nlp2, zeros(16))
(I2,J2,V2) = hess_coo(nlp2, zeros(16))

@test (I2,J2,V2) == (_I,_J,_V)
@test _HH == hess(nlp, nlp.meta.x0)[3:18,3:18]

@test hessian_test_functions(nlp)

@test jacobian_test_functions(nlp)
