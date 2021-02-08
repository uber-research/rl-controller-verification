load("@brezel//brezel/tools/notebook:image.bzl", "notebook_image")

# Guidelines: https://hscc.acm.org/2021/repeatability-evaluation/
#
# Download data folder with:
#   gsutil -m cp -r gs://atcp-data/experiments/quadcopter/rep/data notebooks/
#
notebook_image(
    name = "notebook",
    args = ["--notebook-dir", "papers/hscc_2021/rep/notebooks"],
    deps = ["//papers/hscc_2021/rep/notebooks/python_scripts"],
    data = glob(["notebooks/*", "notebooks/data/**"]),
    imports = ["notebooks/python_scripts"],
)
