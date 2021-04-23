# PDENLPModels

[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Build Status](https://travis-ci.org/tmigot/PDENLPModels.jl.svg?branch=master)](https://travis-ci.org/tmigot/PDENLPModels.jl)
[![Coverage Status](https://coveralls.io/repos/github/tmigot/PDENLPModels.jl/badge.svg?branch=main)](https://coveralls.io/github/tmigot/PDENLPModels.jl?branch=master)
[![codecov.io](http://codecov.io/github/tmigot/PDENLPModels.jl/coverage.svg?branch=master)](http://codecov.io/github/tmigot/PDENLPModels.jl?branch=master)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://tmigot.github.io/PDENLPModels.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://tmigot.github.io/PDENLPModels.jl/dev)

PDENLPModels specializes the [NLPModel API](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) to optimization problem with partial differential equation in the constraints. The package relies on [Gridap.jl](https://github.com/gridap/Gridap.jl) for the modeling and the computation of the derivatives. Find tutorials for using Gridap [here](https://github.com/gridap/Tutorials).

We consider optimization problems of the form:
```math
Find functions (y,u): Y -> ℜⁿ x ℜⁿ and κ ∈ ℜⁿ satisfying

min      ∫_Ω​ f(κ,y,u) dΩ​
s.t.     y solution of a PDE(κ,u)=0
         lcon <= c(κ,y,u) <= ucon
         lvar <= (κ,y,u)  <= uvar
```

## Installation

```
] add github.com/tmigot/PDENLPModels.jl 
```
The current version of PDENLPModels relies on Gridap v0.14.

## Example

```math
min_{y ∈ H^1_0,u ∈ H^1}   0.5 ∫_Ω​ |y(x) - yd(x)|^2dx + 0.5 * α * ∫_Ω​ |u|^2
 s.t.         -Δy = u + h,   for    x ∈  Ω
               y  = 0,       for    x ∈ ∂Ω
where yd(x) = -x[1]^2, h(x) = 1 and α = 1e-2.
```

```julia
using Gridap, PDENLPModels

  # Definition of the domain
  n = 100
  domain = (-1, 1, -1, 1)
  partition = (n, n)
  model = CartesianDiscreteModel(domain, partition)

  # Definition of the spaces
  Xpde = TestFESpace(
    reffe=:Lagrangian, 
    conformity=:H1, 
    valuetype=Float64, 
    model=model, 
    order=2, 
    dirichlet_tags="boundary"
  )
  Ypde = TrialFESpace(Xpde, 0.0)
  Xcon = TestFESpace(
    reffe=:Lagrangian, 
    order=1, 
    valuetype=Float64,
    conformity=:H1, 
    model=model
  )
  Ycon = TrialFESpace(Xcon)

  #Integration machinery
  trian = Triangulation(model)
  degree = 1
  quad = CellQuadrature(trian,degree)

  #Definition of the objective function:
  yd(x) = -x[1]^2
  α = 1e-2
  function f(yu)
    y, u = yu
    0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u
  end

  #Definition of the constraint operator
  function res(yu, v)
    y, u = yu
    ∇(v)⊙∇(y) - v*u
  end
  h(x) = 1.0
  t_Ω = AffineFETerm(res, v -> v * h, trian, quad)
  op  = AffineFEOperator(Y, Xpde, t_Ω)

  nlp = GridapPDENLPModel(f, trian, quad, Ypde, Ycon, Xpde, Xcon, op, name = "Control elastic membrane")
```

## References

> Badia, S., & Verdugo, F.
> Gridap: An extensible Finite Element toolbox in Julia.
> Journal of Open Source Software, 5(52), 2520 (2020).
> [10.21105/joss.02520](https://doi.org/10.21105/joss.02520)