# PDENLPModels.jl Documentation

PDENLPModels specializes the [NLPModel API](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) to optimization problem with partial differential equation in the constraints. The package relies on [Gridap.jl](https://github.com/gridap/Gridap.jl) for the modeling and the computation of the derivatives.

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
] add github.com/tmigot/PDENLPModels.jl#master 
```
The current version of PDENLPModels relies on Gridap v0.14.

## Examples

```@contents
```

We refer to the folder `test/problems` for more examples of problems of different types: calculus of variations, optimal control problem, PDE-constrained problems, and mixed PDE-contrained problems with both function and vector unknowns. Without objective function, the problem reduces to a classical PDE and we refer to [Gridap tutorials](https://github.com/gridap/Tutorials) for examples.


## References

https://github.com/gridap/Gridap.jl
Badia, S., & Verdugo, F. (2020). Gridap: An extensible Finite Element toolbox in Julia.
Journal of Open Source Software, 5(52), 2520.


https://github.com/JuliaSmoothOptimizers/NLPModels.jl
D. Orban and A. S. Siqueira and {contributors} (2020). NLPModels.jl: Data Structures for Optimization Models
