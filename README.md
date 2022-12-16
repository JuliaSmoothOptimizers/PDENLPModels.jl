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
[doi-img]: https://joss.theoj.org/papers/10.21105/joss.04736/status.svg
[doi-url]: https://doi.org/10.21105/joss.04736

PDENLPModels specializes the [NLPModel API](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) to optimization problems with partial differential equations in the constraints. The package relies on [Gridap.jl](https://github.com/gridap/Gridap.jl) for the modeling and the computation of the derivatives. Find tutorials for using Gridap [here](https://github.com/gridap/Tutorials).

We consider optimization problems of the form:
```math
Find functions (y,u): Y -> ℜⁿ x ℜⁿ and κ ∈ ℜⁿ satisfying

min      ∫_Ω​ f(κ,y,u) dΩ​
s.t.     y solution of a PDE(κ,u)=0
         lcon <= c(κ,y,u) <= ucon
         lvar <= (κ,y,u)  <= uvar
```

We refer to the the repository [PDEOptimizationProblems](https://github.com/tmigot/PDEOptimizationProblems) for examples of problems of different types: calculus of variations, optimal control problem, PDE-constrained problems, and mixed PDE-contrained problems with both function and algebraic unknowns.

## Installation

```
] add PDENLPModels
```
The current version of PDENLPModels relies on Gridap v0.15.5.

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

  # Definition of the spaces:
  valuetype = Float64
  reffe = ReferenceFE(lagrangian, valuetype, 2)
  Xpde = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
  y0(x) = 0.0
  Ypde = TrialFESpace(Xpde, y0)

  reffe_con = ReferenceFE(lagrangian, valuetype, 1)
  Xcon = TestFESpace(model, reffe_con; conformity = :H1)
  Ycon = TrialFESpace(Xcon)

  # Integration machinery
  trian = Triangulation(model)
  degree = 1
  dΩ = Measure(trian, degree)

  # Objective function:
  yd(x) = -x[1]^2
  α = 1e-2
  function f(y, u)
    ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u) * dΩ
  end

  # Definition of the constraint operator
  ω = π - 1 / 8
  h(x) = -sin(ω * x[1]) * sin(ω * x[2])
  function res(y, u, v)
    ∫(∇(v) ⊙ ∇(y) - v * u - v * h) * dΩ
  end

  # initial guess
  npde = num_free_dofs(Ypde)
  ncon = num_free_dofs(Ycon)
  xin = zeros(npde + ncon)

  nlp = GridapPDENLPModel(xin, f, trian, Ypde, Ycon, Xpde, Xcon, res, name = "Control elastic membrane")
```

## References

> Badia, S., & Verdugo, F.
> Gridap: An extensible Finite Element toolbox in Julia.
> Journal of Open Source Software, 5(52), 2520 (2020).
> [10.21105/joss.02520](https://doi.org/10.21105/joss.02520)

## How to Cite

If you use PDENLPModels.jl in your work, please cite using the format given in [CITATION.cff](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl/blob/main/CITATION.cff).

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
