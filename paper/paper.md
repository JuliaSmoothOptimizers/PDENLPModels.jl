---
title: 'PDENLPModels.jl: An NLPModel API for Optimization Problems with PDE-Constraints'
tags:
  - Julia
  - nonlinear optimization
  - large-scale optimization
  - constrained optimization
  - nonlinear programming
  - PDE-constrained optimization
authors:
  - name: Tangi Migot^[corresponding author]
    orcid: 0000-0001-7729-2513
    affiliation: 1
  - name: Dominique Orban
    orcid: 0000-0002-8017-7687
    affiliation: 1
  - name: Abel Soares Siqueira
    orcid: 0000-0003-4451-281X
    affiliation: 2
affiliations:
 - name: GERAD and Department of Mathematics and Industrial Engineering, Polytechnique Montréal, QC, Canada.
   index: 1
 - name: Netherlands eScience Center, Amsterdam, NL
   index: 2
date: 31 August 2022
bibliography: paper.bib

---

# Summary

Shape optimization, optimal control, and parameter estimation of systems governed by partial differential equations (PDE) give rise to a class of problems known as PDE-constrained optimization [@hinze2008optimization].
\texttt{PDENLPModels.jl} is a Julia [@bezanson2017julia] package for modeling and discretizing optimization problems with mixed algebraic and PDE in the constraints. 
The general form of the problems over some domain $\Omega \subset \mathbb{R}^d$ is
\begin{equation*}
  \begin{array}{lll}
    \underset{y, u, \theta}{\text{minimize}} \int_\Omega J(y, u, \theta)d\Omega \ \mbox{ subject to} & e(y, u, \theta) = 0, & \mbox{(governing PDE on $\Omega$)} \\
    & l_{yu} \leq (y, u) \leq u_{yu}, & \mbox{(functional bound constraints)} \\
    & l_{\theta} \leq \theta \leq u_{\theta}, & \mbox{(bound constraints)}
	\end{array}
\end{equation*}

where $y : \Omega \rightarrow \mathcal{Y}$ is the state, $u : \Omega \rightarrow \mathcal{U}$ is the control, and $\theta \in \mathbb{R}^k$ are algebraic variables. $J:\mathcal{Y} \times \mathcal{U} \times \mathbb{R}^k \rightarrow \mathbb{R}$ and $e : \mathcal{Y} \times \mathcal{U} \times \mathbb{R}^k \rightarrow \mathcal{C}$ are smooth mappings. $(\mathcal{Y},\| \cdot \|_{\mathcal{Y}})$, $(\mathcal{U},\| \cdot \|_{\mathcal{U}})$, and $(\mathcal{C},\| \cdot \|_{\mathcal{C}})$ are real Banach spaces, $l_{\theta}, u_{\theta} \in \mathbb{R}^k$ are bounds on $\theta$, and $l_{yu}, u_{yu}:\Omega \rightarrow \mathcal{Y} \times \mathcal{U}$ are functional bounds on the controls and states.

After discretization of the domain $\Omega$, the integral, and the derivatives, the resulting problem is a nonlinear optimization problem of the form
\begin{equation*}
    \underset{x \in \mathbb{R}^{N_y + N_u + N_\theta}}{\text{minimize}} \quad f(x) \quad \text{subject to} \quad c(x) = 0, \quad l \leq x \leq u,
\end{equation*}
where $l, u \in \mathbb{R}^{N_y + N_u + N_\theta}$, $f:\mathbb{R}^{N_y} \times \mathbb{R}^{N_u} \times \mathbb{R}^{N_\theta} \rightarrow \mathbb{R}$ and $c:\mathbb{R}^{N_y} \times \mathbb{R}^{N_u} \times \mathbb{R}^{N_\theta} \rightarrow \mathbb{R}^{N_y}$.

The two main challenges in modeling such a problem are to be able to (i) discretize the domain and generate corresponding discretizations of the objective and constraints, and (ii) evaluate derivatives of $f$ and $c$ with respect to all variables.
Several packages allow the user to define the domain, meshes, function spaces, and finite-element families to approximate unknowns, and to model functionals and sets of PDEs in a weak form using powerful high-level notation. The main ones are \texttt{FEniCS.jl}, a wrapper for the FEniCS library [@logg2012automated], \texttt{Ferrite.jl} [@Carlsson_Ferrite_jl_2021], \texttt{FinEtools.jl} [@Krysl_FinEtools_jl_2021], \texttt{JuliaFEM.jl} [@JuliaFEM] [@JuliaFEM-paper], and \texttt{Gridap.jl} [@badia2020gridap] [@verdugo2022software].
In \texttt{PDENLPModels.jl}, we focus on the latter as it is exclusively written in Julia and supports a variety of discretizations and meshing possibilities.
Additionally, \texttt{Gridap.jl} has an expressive API allowing one to model complex PDEs with few lines of code, and to write the underlying weak form with a syntax that is almost one-to-one with mathematical notation.

However, the above packages are designed for sets of PDEs and not for optimization, so that only the derivatives of the PDE with respect to $y$ can be evaluated. 
In addition, inequalities are not supported.
\texttt{PDENLPModels.jl} extends \texttt{Gridap.jl}'s differentiation facilities to also obtain derivatives useful for optimization, i.e., first and second derivatives of the objective and constraint functions with respect to controls and finite-dimensional variables.
Because we aim to solve nonconvex optimization problems with inequality constraints, it would not be appropriate, or even feasible, to solve a system of PDEs representing the optimality conditions with \texttt{Gridap.jl}.

\texttt{PDENLPModels.jl} exports the `GridapPDENLPModel` type, which uses \texttt{Gridap.jl} for the discretization of the functional spaces by finite-elements. The resulting model is an instance of an `AbstractNLPModel`, as defined in \texttt{NLPModels.jl} [@orban-siqueira-nlpmodels-2020], that provides access to objective and constraint function values, to their first and second derivatives, and to any information that a solver might request from a model. 

The role of \texttt{NLPModels.jl} is to define an API that users and solvers can rely on. It is the role of other packages to implement facilities that create models compliant with the NLPModels API. There are several examples of this in JuliaSmoothOptimizers organization: \texttt{AmplNLReader.jl} [@orban-siqueira-amplnlreader-2020] allows users to connect the AMPL modeling language with NLPModels, \texttt{CUTEst.jl} [@orban-siqueira-cutest-2020] does the same for the SIF language, \texttt{NLPModelsJuMP.jl} [@montoison-orban-siquiera-nlpmodelsjump-2020] does the same with JuMP models, etc. In those three examples, there exist underlying modeling tools (in Julia or not). \texttt{PDENLPModels.jl} is different in that there is no existing generic interface for optimization with PDEs. Instead, \texttt{PDENLPModels.jl} interacts with \texttt{Gridap.jl} to evaluate functionals and differential operators based on a discretization. \texttt{Gridap.jl} in itself does not let users model optimization problems; only systems of PDEs. \texttt{PDENLPModels.jl} provides all the extra facilities for users and solvers to interact with a PDE-constrained optimization problem as they would with a JuMP model, an AMPL model, or any other model that complies with the NLPModels API.
As such, \texttt{PDENLPModels.jl} offers an interface between generic PDE-constrained optimization problems and cutting-edge optimization solvers such as \texttt{Artelys Knitro} [@byrd2006k] via \texttt{NLPModelsKnitro.jl} [@orban-siqueira-nlpmodelsknitro-2020], \texttt{Ipopt} [@wachter2006implementation] via \texttt{NLPModelsIpopt.jl} [@orban-siqueira-nlpmodelsipopt-2020], \texttt{DCISolver.jl} [@migot2022dcisolver], \texttt{Percival.jl} [@percival-jl], and any solver accepting an `AbstractNLPModel` as input, see JuliaSmoothOptimizers (JSO) [@jso].

The following example shows how to solve a Poisson control problem with Dirichlet boundary conditions using \texttt{DCISolver.jl}:
\begin{equation*}
  \begin{array}{lll}
    \underset{y, u}{\text{minimize}} \int_{(-1,1)^2} \frac{1}{2}\|y_d - y\|^2 +\frac{\alpha}{2}\|u\|^2 d\Omega \quad \mbox{subject to} & \Delta y - u - h = 0, & \mbox{on } \Omega.\\
    & y = 0, & \mbox{on } \partial\Omega,
  \end{array}
\end{equation*}
for some given functions $y_d, h:(-1,1)^2 \rightarrow \mathbb{R}$, and $\alpha > 0$.

![](code.pdf)

# Statement of need

For PDEs, there are five main ways to discretize functions and their derivatives:

- Finite-difference methods: functions are represented on a grid, e.g., \texttt{DiffEqOperators.jl} [@DifferentialEquations.jl-2017] or Trixi.jl [@schlottkelakemper2020trixi];
- Finite-volume methods: functions are represented by a discretization of their integral;
- Spectral methods: functions are expanded in a global basis, e.g., \texttt{FFTW.jl} [@FFTW.jl-2005] and \texttt{ApproxFun.jl} [@ApproxFun.jl-2014];
- Physics-informed neural networks: functions are represented by neural networks, e.g., \texttt{NeuralPDE.jl} [@zubov2021neuralpde];
- Finite-element methods: functions are expanded in a local basis.

With finite-elements discretization, it is easy to increase the order of the elements or locally refine the mesh so that the physical fields can be approximated accurately. Another advantage is that you can straightforwardly combine different kinds of approximation functions, leading to mixed formulations. Finally, curved or irregular geometries of the domain are handled in a natural way.

Outside of Julia, there exist libraries handling finite-elements methods such as \texttt{deal.II} [@bangerth2007deal], \texttt{FEniCS} [@logg2012automated], \texttt{PETSc} [@petsc-user-ref], and \texttt{FreeFEM++} [@hecht2012new]. There exists a Julia wrapper to \texttt{FEniCS} [@DifferentialEquations.jl-2017] and \texttt{PETSc} [@ptesc-julia]. However, interfaces to low-level libraries have limitations that pure Julia implementations do not have, including the ability to generate models with various arithmetic types.

Julia’s JIT compiler is attractive for the design of efficient scientific computing software, and, in particular, mathematical optimization [@lubin2015computing], and has become a natural choice for developing new modeling tools. There are other packages available in Julia for optimization problems with PDE in the constraints. \texttt{jInv.jl} [@ruthotto2017jinv] and \texttt{ADCME.jl} [@xu2020adcme] focus on inverse problems. \texttt{DifferentialEquations.jl} [@DifferentialEquations.jl-2017] is a suite for numerically solving differential equations written in Julia, which includes features for parameter estimation and Bayesian analysis. \texttt{InfiniteOpt.jl} [@pulsipher2022unifying] provides a general mathematical abstraction to express and solve infinite-dimensional optimization problems including with PDEs in the constraints handled by finite-differences. \texttt{TopOpt.jl} [@huang2021topoptjl] is a package for topology optimization.
However, to the best of our knowledge, there are no packages with the generality of \texttt{PDENLPModels.jl}.

Optimization problems with PDEs in the constraints have been in the spotlight in recent years as
challenging and highly structured. 
The great divide between optimization libraries and PDE libraries makes it difficult for optimization research to benefit
from testing on a large base of PDE-constrained problems and PDE libraries
to benefit from the latest advances in optimization.
\texttt{PDENLPModels.jl} fills this gap by providing generic discretized models that can be solved by any solver from JuliaSmoothOptimizers.

# Acknowledgements

Tangi Migot is supported by IVADO and the Canada First Research Excellence Fund / Apogée,
and Dominique Orban is partially supported by an NSERC Discovery Grant.

# References
