using Documenter
using PDENLPModels

makedocs(
    sitename = "PDENLPModels",
    format = Documenter.HTML(),
    modules = [PDENLPModels],
    pages = [
    "index.md",
    "Using Krylov.jl in Gridap" => "linear_pde_solver_Krylov.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#

#Tanj: to visualize locally
#julia -e 'using LiveServer; serve(dir="docs/build")'
