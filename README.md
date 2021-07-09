# PDENLPModels

| **Documentation** | **CI** | **Coverage** | **Release** | **DOI** |
|:-----------------:|:------:|:------------:|:-----------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-ci][build-ci-img]][build-ci-url] | [![codecov][codecov-img]][codecov-url] | [![release][release-img]][release-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/PDENLPModels.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/PDENLPModels.jl/dev
[build-ci-img]: https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl/workflows/CI/badge.svg?branch=main
[build-ci-url]: https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl/actions
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/PDENLPModels.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaSmoothOptimizers/PDENLPModels.jl
[release-img]: https://img.shields.io/github/v/release/JuliaSmoothOptimizers/PDENLPModels.jl.svg?style=flat-square
[release-url]: https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl/releases
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.5056629.svg
[doi-url]: https://doi.org/10.5281/zenodo.5056629

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
] add github.com/JuliaSmoothOptimizers/PDENLPModels.jl 
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

## How to Cite

If you use PDENLPModels.jl in your work, please cite using the format given in [CITATION.bib](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl/blob/main/CITATION.bib).