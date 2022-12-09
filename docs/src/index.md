# PDENLPModels.jl Documentation

PDENLPModels is a Julia package that specializes the [NLPModel API](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) for modeling and discretizing optimization problems with mixed algebraic and PDE in the constraints.

We consider optimization problems of the form: find functions (y,u) and κ ∈ ℜⁿ satisfying
```math
\left\lbrace
\begin{aligned}
\min\limits_{κ,y,u} \  & \int_{\Omega} f(κ,y,u)dx, \\
\text{ s.t. } & \text{y solution of a PDE(κ,u)},\\
                  & lcon \leq c(κ,y,u) \leq ucon,\\
                  & lvar \leq (κ,y,u)  \leq uvar.\\
\end{aligned}
\right.
```

The main challenges in modeling such a problem are to be able to discretize the domain and generate corresponding discretizations of the objective and constraints, and their evaluate derivatives with respect to all variables.
We use [Gridap.jl](https://github.com/gridap/Gridap.jl) to define the domain, meshes, function spaces, and finite-element families to approximate unknowns, and to model functionals and sets of PDEs in a weak form. 
PDENLPModels extends [Gridap.jl](https://github.com/gridap/Gridap.jl)'s differentiation facilities to also obtain derivatives useful for optimization, i.e., first and second derivatives of the objective and constraint functions with respect to controls and finite-dimensional variables.

After discretization of the domain $\Omega$, the integral, and the derivatives, the resulting problem is a nonlinear optimization problem.
PDENLPModels exports the `GridapPDENLPModel` type, an instance of an `AbstractNLPModel`, as defined in [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl), which provides access to objective and constraint function values, to their first and second derivatives, and to any information that a solver might request from a model. 
The role of [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) is to define an API that users and solvers can rely on. It is the role of other packages to implement facilities that create models compliant with the NLPModels API. We refer to [juliasmoothoptimizers.github.io](https://juliasmoothoptimizers.github.io) for tutorials on the NLPModel API.

As such, PDENLPModels offers an interface between generic PDE-constrained optimization problems and cutting-edge optimization solvers such as Artelys Knitro via [NLPModelsKnitro.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsKnitro.jl), Ipopt via [NLPModelsIpopt.jl](https://github.com/JuliaSmoothOptimizers/NLPModelsIpopt.jl) , [DCISolver.jl](https://github.com/JuliaSmoothOptimizers/DCISolver.jl), [Percival.jl](https://github.com/JuliaSmoothOptimizers/Percival.jl), and any solver accepting an `AbstractNLPModel` as input, see [JuliaSmoothOptimizers](https://juliasmoothoptimizers.github.io).

## Installation

```
] add PDENLPModels
```
The current version of PDENLPModels relies on Gridap v0.15.5.

## Table of Contents

```@contents
```

## Examples

You can also check the tutorial [Solve a PDE-constrained optimization problem](https://jso-docs.github.io/solve-pdenlpmodels-with-jsosolvers/) on our site, [juliasmoothoptimizers.github.io](https://juliasmoothoptimizers.github.io).

We refer to the folder `test/problems` for more examples of problems of different types: calculus of variations, optimal control problem, PDE-constrained problems, and mixed PDE-contrained problems with both function and vector unknowns. An alternative is to visit the repository [PDEOptimizationProblems](https://github.com/tmigot/PDEOptimizationProblems) that contains a collection of test problems. Without objective function, the problem reduces to a classical PDE and we refer to [Gridap tutorials](https://github.com/gridap/Tutorials) for examples.

## References

[Gridap.jl](https://github.com/gridap/Gridap.jl)
Badia, S., Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia.
Journal of Open Source Software, 5(52), 2520.


[NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)
D. Orban, A. S. Siqueira and contributors (2020). NLPModels.jl: Data Structures for Optimization Models

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
