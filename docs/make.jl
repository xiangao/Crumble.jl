using Documenter
using Crumble

makedocs(
    sitename = "Crumble.jl",
    modules = [Crumble],
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/01_getting_started.md",
            "Main Vignette" => "tutorials/02_main_vignette.md",
        ],
        "Reference" => "reference.md",
    ],
    warnonly = true,
)
