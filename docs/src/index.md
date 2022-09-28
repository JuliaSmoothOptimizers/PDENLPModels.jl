# PDENLPModels.jl Documentation

PDENLPModels specializes the [NLPModel API](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) to optimization problems with partial differential equations in the constraints. The package relies on [Gridap.jl](https://github.com/gridap/Gridap.jl) for the modeling and the computation of the derivatives.

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

## Installation

```
] add PDENLPModels
```
The current version of PDENLPModels relies on Gridap v0.15.5.

## Examples

```@contents
```

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
