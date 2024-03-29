using Documenter
using Printf
using PDENLPModels

# Add index.md file as introduction to navigation menu
pages = [
  "Introduction" => "index.md",
  "Calculus of Variations" => "tore.md",
  "PDE-constrained optimization" => "poisson-boltzman.md",
  "Reference" => "reference.md",
]

makedocs(
  sitename = "PDENLPModels.jl",
  strict = true,
  format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
  modules = [PDENLPModels],
  pages = pages,
)

deploydocs(
  repo = "github.com/JuliaSmoothOptimizers/PDENLPModels.jl.git",
  push_preview = true,
  devbranch = "main",
)
