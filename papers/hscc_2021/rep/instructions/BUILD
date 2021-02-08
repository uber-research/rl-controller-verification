load("@ulatex//:latex.bzl", "latex_document")
latex_document(
    name = "instructions",
    main = "main.tex",
    srcs = ["main.bib"],
    data = glob(["*.png"]),
    deps = ["@ulatex//texmf", "uLatex.cls"],
)
