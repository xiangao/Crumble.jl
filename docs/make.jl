using Documenter
using Crumble

makedocs(
    sitename = "Crumble.jl",
    modules = [Crumble],
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting started" => "tutorials/01_getting_started.md",
            "Interventional effects" => "tutorials/02_main_vignette.md",
        ],
        "Reference" => "reference.md",
    ],
    warnonly = true,
)
