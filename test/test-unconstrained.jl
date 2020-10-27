using Gridap, LinearAlgebra, NLPModels, SparseArrays, Test

#include("../GridapPDENLPModel.jl")
using Main.PDENLPModels

ubis(x) =  x[1]^2+x[2]^2
f(y,u) = 0.5 * (ubis - u) * (ubis - u) + 0.5 * y * y
function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
    y, u = yu
    f(y,u)
end

domain = (0,1,0,1)
n = 2^7
partition = (n,n)
@time model = CartesianDiscreteModel(domain,partition)

order = 1
@time V0 = TestFESpace(reffe=:Lagrangian, order=order, valuetype=Float64,
                       conformity=:H1, model=model, dirichlet_tags="boundary")
u(x) = 0.0
U    = TrialFESpace(V0,u)

Yedp = MultiFieldFESpace([U])
Xedp = V0
Ycon = MultiFieldFESpace([U])
Xcon = V0

trian = Triangulation(model)
degree = 2
@time quad = CellQuadrature(trian,degree)

function res(yu, v) #y is the solution of the PDE and u the parameter
 y, u = yu
 ∇(v)⊙∇(y) - v*u
end
t_Ω = FETerm(res,trian,quad)
Y = MultiFieldFESpace(vcat(Yedp.spaces, Ycon.spaces))
#FEOperator(trial::FESpace,test::FESpace,assem::Assembler,terms::FETerm...)
op = FEOperator(Y, Xedp, t_Ω)

xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
@time nlp = GridapPDENLPModel(xin, zeros(0), f, Yedp, Ycon, Xedp, Xcon, trian, quad)

sf = Int(nlp.meta.nvar/2)
x1 = vcat(rand(sf), ones(sf)); x=x1;v=x1;

@time fx = obj(nlp, x1);
@time gx = grad(nlp, x1);
@time _fx, _gx = objgrad(nlp, x1);
@test norm(gx - _gx) <= eps(Float64)
@test norm(fx - _fx) <= eps(Float64)

@time  Hx = hess(nlp, x1);
@time _Hx = hess(nlp, rand(nlp.meta.nvar))
@test norm(Hx - _Hx) <= eps(Float64) #the hesian is constant
@time Hxv = Hx * v;
@time _Hxv = hprod(nlp, x1, v);
norm(Hxv - _Hxv)

#Check the solution:
cell_xs = get_cell_coordinates(trian)
midpoint(xs) = sum(xs)/length(xs)
cell_xm = apply(midpoint, cell_xs) #this is a vector of size num_cells(trian)
cell_ubis = apply(ubis, cell_xm) #this is a vector of size num_cells(trian)
solu = get_free_values(Gridap.FESpaces.interpolate(U, cell_ubis))
soly = get_free_values(zero(U))
sol = vcat(soly, solu)

@test obj(nlp, sol) <= 1/n
@test norm(grad(nlp, sol)) <= 1/n

using JSOSolvers
@time _t = lbfgs(nlp, x = x1, rtol = 0.0, atol = 1e-10) #lbfgs modifies the initial point !!
@test norm(_t.solution[1:Int(nlp.meta.nvar/2)] - soly, Inf) <= 1/n # <= sqrt(eps(Float64))
norm(_t.solution[Int(nlp.meta.nvar/2) + 1: nlp.meta.nvar] - solu, Inf)
#"On devrait faire un print de la deuxième partie de la solution pour comparer avec ubis?"
