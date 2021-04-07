using Documenter
using Literate
using Printf
using PDENLPModels

# Add index.md file as introduction to navigation menu
pages = ["Introduction" => "index.md",
         "PDENLPModels Progress" => "nlpmodels.md",
         "Krylov.jl to solve linear PDE" => "KrylovforLinearPDE.md"]

makedocs(
    sitename = "PDENLPModels.jl",
    format = Documenter.HTML(),
    modules = [PDENLPModels],
    pages = pages
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/tmigot/PDENLPModels.jl.git"
)

#Tanj: to visualize locally
#julia -e 'using LiveServer; serve(dir="docs/build")'
