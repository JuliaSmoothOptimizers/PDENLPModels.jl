using Documenter

if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

#Documenter.post_status(; type="pending", repo="github.com/tmigot/PDENLPModels.jl.git")

using Literate
using Printf
using PDENLPModels

pages_dir = joinpath(@__DIR__,"src","pages")
notebooks_dir = joinpath(@__DIR__,"src","notebooks")

Sys.rm(pages_dir;recursive=true,force=true)
Sys.rm(notebooks_dir;recursive=true,force=true)

repo_src = joinpath(@__DIR__,"src")
repo_pluto = joinpath(@__DIR__,"..","pluto")
repo_jsopde = joinpath(@__DIR__,"..","JSOPDESolver")

# Add index.md file as introduction to navigation menu
pages = ["Introduction"=> "index.md"]

#########################################################################
# TEMP
i = 1
title    = "Solve a PDE with Gridap.jl and Krylov.jl"
filename = "linear_pde_solver_Krylov.jl"
tutorial_prefix = string("t",@sprintf "%03d_" i)
tutorial_title = string("# # Tutorial ", i, ": ", title)
tutorial_file = string(tutorial_prefix,splitext(filename)[1])
# Generate notebooks
tutorial_title = string("# # Tutorial ", i, ": ", title)
function preprocess_notebook(content)
  return string(tutorial_title, "\n\n", content)
end
Literate.notebook(joinpath(repo_jsopde,filename), notebooks_dir;
                  name=filename, preprocess=preprocess_notebook,
                  documenter=false, execute=false)

# Generate markdown
function preprocess_docs(content)
  #return string(tutorial_title, "\n", binder_badge, "\n", nbviwer_badge, "\n\n", content)
  return string(tutorial_title,  "\n\n", content)
end
Literate.markdown(joinpath(repo_jsopde,filename), pages_dir;
                  name=tutorial_file, preprocess=preprocess_docs,
                  codefence="```julia" => "```")
# Generate navigation menu entries
ordered_title = string(i, " ", title)
path_to_markdown_file = joinpath("pages",string(tutorial_file,".md"))
push!(pages, (ordered_title=>path_to_markdown_file))
#END TEMP
#########################################################################

makedocs(
    sitename = "PDENLPModels",
    format = Documenter.HTML(),
    modules = [PDENLPModels],
    pages = pages
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/tmigot/PDENLPModels.jl",#.git
    devbranch = "master",
    devurl = "dev",
    versions = ["dev" => "dev"]
)

#Tanj: to visualize locally
#julia -e 'using LiveServer; serve(dir="docs/build")'
