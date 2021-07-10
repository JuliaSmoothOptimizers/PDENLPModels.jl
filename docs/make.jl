using Documenter
using Printf
using PDENLPModels

# Add index.md file as introduction to navigation menu
pages = [
  "Introduction" => "index.md",
  "Calculus of Variations" => "tore.md",
  "PDE-constrained optimization" => "poisson-boltzman.md",
  "Krylov.jl to solve linear PDE" => "KrylovforLinearPDE.md",
  "Reference" => "reference.md",
]

makedocs(
  sitename = "PDENLPModels.jl",
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  modules = [PDENLPModels],
  pages = pages,
)

deploydocs(repo = "github.com/JuliaSmoothOptimizers/PDENLPModels.jl.git", push_preview = true, devbranch = "main")
