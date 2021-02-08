load("@ulatex//:latex.bzl", "latex_document")
latex_document(
    name = "paper",
    main = "main.tex",
    srcs = ["main.bib"],
    data = glob(["data/*.csv", "pictures/*.png"]),
    deps = ["acmart.cls", "extra-stuff.sty"],
)
