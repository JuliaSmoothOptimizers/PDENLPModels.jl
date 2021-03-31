using Plots, LinearAlgebra, SparseArrays
###############################################################
#Data for SIS
x0 = [0.6, 0.1] #I, S, R
N = sum(x0)
#=
a = 0.2
b = 0.1 #0.1 ou 0.7
μ = 0.1 #vaccination rate :)
# The usual SIR:
@warn "Add time dependence"
F(x) = vcat(a .* x[1:n] .* x[n+1:2*n] .- b .* x[1:n], 
            -c .- a .* x[1:n] .* x[n+1:2*n],
            b .* x[2 * n + 1: 3 * n] .+ c
            )
=#
T = 7 #final time
###############################################################
#Now we discretize by hand with forward finite differences
n = 10
h = T/n

#=
AI = 1/h * Bidiagonal(ones(n), -ones(n-1), :L)
AS = 1/h * Bidiagonal(ones(n), -ones(n-1), :L)
A0 = zeros(2 * n); A0[1] = -x0[1] / h; A0[n+1] = -x0[2] / h;

c(x) = vcat(AI * x[1:n], AS * x[n+1:2*n]) + A0 - F(x)
=#
################################################################
# The exact solution of the ODE is given by:
#known in Exact Solution to a Dynamic SIR Model, M. Bohner, S. Streipert, D. F. M. Torres, Researchgate preprint 2018
# if b(t)=1/(t+1) and c(t)=2/(t+1)
function solI(t)
    κ = x0[1]/x0[2]
    ρ = (κ + 1) * (t + 1)
    I = x0[1] * (κ + 1 + t)/((κ + 1) * (t + 1)^2)
    S = x0[2] * (κ + 1 + t)/((κ + 1) * (t + 1))
    R = N - (κ + 1 + t)/(t+1) * ( x0[2]/(κ + 1) - x0[1]/ρ )
    return I
end

function solS(t)
    κ = x0[1]/x0[2]
    ρ = (κ + 1) * (t + 1)
    I = x0[1] * (κ + 1 + t)/((κ + 1) * (t + 1)^2)
    S = x0[2] * (κ + 1 + t)/((κ + 1) * (t + 1))
    R = N - (κ + 1 + t)/(t+1) * ( x0[2]/(κ + 1) - x0[1]/ρ )
    return S
end
################################################################
# Some checks and plots
# Vectorized solution
sol_Ih = [solI(t) for t=h:h:T]
sol_Sh = [solS(t) for t=h:h:T]

#@show norm(c(vcat(sol_Ih, sol_Sh)), Inf) #check the discretization by hand

plot(0:h:T, vcat(x0[1], sol_Ih))
plot!(0:h:T, vcat(x0[2], sol_Sh))

png("test")

################################################################
# Using Gridap and PDENLPModels
using Gridap, PDENLPModels

function sis_gridap(x0, n, T)
    kp(x) = 1.01
    kr(x) = 2.03

    model = CartesianDiscreteModel((0,T),n)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"diri0",[1]) #initial time condition

    #=
    V = TestFESpace(
        reffe=:Lagrangian, conformity=:H1, valuetype=VectorValue{2,Float64},
        model=model, labels=labels, order=1, dirichlet_tags=["diri0"])
    uD0 = VectorValue(x0[1], x0[2])  
    U = TrialFESpace(V,uD0)
    =#

    Vcon = TestFESpace(
            reffe=:Lagrangian, order=1, valuetype=Float64,
            conformity=:L2, model=model)
    Ucon = TrialFESpace(Vcon)
    Xcon = MultiFieldFESpace([Vcon])
    Ycon = MultiFieldFESpace([Ucon])

    function f(yu) #:: Union{Gridap.MultiField.MultiFieldFEFunction, Gridap.CellData.GenericCellField}
        cf, pf, uf = yu
        kp * pf
    end

    #If we rewrite it as one? and then split yu = bf, cf
    VI = TestFESpace(
        reffe=:Lagrangian, conformity=:H1, valuetype=Float64,
        model=model, labels=labels, order=1, dirichlet_tags=["diri0"]) 
    UI = TrialFESpace(VI, x0[1])
    VS = TestFESpace(
        reffe=:Lagrangian, conformity=:H1, valuetype=Float64,
        model=model, labels=labels, order=1, dirichlet_tags=["diri0"])
    US = TrialFESpace(VS, x0[2])
    Xpde = MultiFieldFESpace([VI, VS])
    Ypde = MultiFieldFESpace([UI, US])

    @law conv(u,∇u) = (∇u ⋅one(∇u))⊙u
    c(u,v) = conv(v,∇(u)) #v⊙conv(u,∇(u))
    function res_pde_nl(yu,v)
        cf, pf, uf = yu
        p, q = v
        #eq. (2) page 3
        c(cf, p) + c(pf, q)
    end
    function res_pde(yu,v)
        cf, pf, uf = yu
        p, q = v
        #eq. (2) page 3
        - p * ( kp * pf * (1. - cf) - kr * cf * (1. - cf - pf) ) + q * ( uf * kr * cf * (1. - cf - pf) - kp * pf * pf )
    end

    trian = Triangulation(model)
    degree = 1
    quad = CellQuadrature(trian,degree)
    t_Ω_nl = FETerm(res_pde_nl,trian,quad)
    t_Ω = FETerm(res_pde,trian,quad)
    Y = MultiFieldFESpace([UI, US, Ucon])
    op_sir = FEOperator(Ypde,Xpde,t_Ω_nl)#,t_Ω) #FEOperator(Ypde,Xpde,t_Ω)

    xin = zeros(Gridap.FESpaces.num_free_dofs(Y))
    nlp = GridapPDENLPModel(xin, f, trian, quad, Ypde, Ycon, Xpde, Xcon, op_sir) 
    return nlp
end
n = 10
nlp = sis_gridap(x0, n, T)
xr = rand(nlp.meta.nvar)
jac(nlp, xr)
################################################################
# Testing:
using NLPModelsTest, Test

atol, rtol = √eps(), √eps()
# check the value at the solution:
#=
kmax = 6 #beyond is tough
for k=1:kmax
    local sol_Ih, sol_Sh, h, n, nlp
    n = 10^k
    nlp = sis_gridap(x0, n, a, b, T)
    h = T/n
    sol_Ih = [solI(t) for t=h:h:T]
    sol_Sh = [solS(t) for t=h:h:T]
    sol_b  = [1/(t+1) for t=0:h:T]
    sol_c  = [2/(t+1) for t=0:h:T]
    sol = vcat(sol_Ih, sol_Sh, sol_b, sol_c)
    res = norm(cons(nlp, sol), Inf)
    @show res
    val = obj(nlp, sol)
    @show val
    if res <= 1e-5 && val <= 1e-10
        @test true
        break
    end
    if k == kmax @test false end
end
=#

n = 10
nlp = sis_gridap(x0, n, T)
xr = rand(nlp.meta.nvar)
#Beta-tests
@test obj(nlp, xr) != nothing
@test grad(nlp, xr) != nothing
@test cons(nlp, xr) != nothing
#@test jac(nlp, xr) != nothing
@test hess(nlp, xr) != nothing
@test hess(nlp, xr, nlp.meta.y0) != nothing

#check derivatives
@test gradient_check(nlp, x = xr, atol = atol, rtol = rtol) == Dict{Tuple{Int64,Int64},Float64}()
#@test jacobian_check(nlp, x = xr, atol = atol, rtol = rtol) == Dict{Tuple{Int64,Int64},Float64}()
#Issue with the jacobian here!!!
ymp = hessian_check(nlp, x = xr, atol = atol, rtol = rtol)
@test !any(x -> x!=Dict{Tuple{Int64,Int64},Float64}(), values(ymp))
#ymp2 = hessian_check_from_grad(nlp, x = xr, atol = atol, rtol = rtol) #uses the jacobian
#@test !any(x -> x!=Dict{Tuple{Int64,Int64},Float64}(), values(ymp2))

#=
#This is not working:
@show "Tanj: quick test"
jac(nlp, nlp.meta.x0)
@show "ouf..."
=#