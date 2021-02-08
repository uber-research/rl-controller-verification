load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")


SO_LIBS = [
    "stl_abs_node.so",
    "stl_addition_node.so",
    "stl_always_node.so",
    "stl_and_node.so",
    "stl_combinatorial_binary_node.so",
    "stl_comp_op.so",
    "stl_constant_node.so",
    "stl_division_node.so",
    "stl_fall_node.so",
    "stl_historically_bounded_node.so",
    "stl_historically_node.so",
    "stl_iff_node.so",
    "stl_implies_node.so",
    "stl_io_type.so",
    "stl_multiplication_node.so",
    "stl_node.so",
    "stl_not_node.so",
    "stl_once_bounded_node.so",
    "stl_once_node.so",
    "stl_operator_type.so",
    "stl_or_node.so",
    "stl_precedes_bounded_node.so",
    "stl_predicate_node.so",
    "stl_rise_node.so",
    "stl_sample.so",
    "stl_since_bounded_node.so",
    "stl_since_node.so",
    "stl_subtraction_node.so",
    "stl_time.so",
    "stl_xor_node.so",
]


def rtamt_libs(name, repo="rtamt_deps"):
    if repo == 'rtamt':
        fail('Setting repo="rtamt" will not work.')

    lib_dir = "$EXT_BUILD_ROOT/external/{}/rtamt/lib".format(repo)
    cmake_external(
        name = name,
        lib_source = "@{}//:all".format(repo),
        cmake_options = ["-GNinja", "-DPythonVersion=3"],
        make_commands = ["ninja"],
        working_directory = "rtamt",
        shared_libraries = SO_LIBS,
        postfix_script = "cp -L -r --no-target-directory {} $INSTALLDIR/lib".format(lib_dir),
        install_prefix = ".",
        out_lib_dir = "lib/rtamt_stl_library_wrapper",
        visibility = ["//visibility:public"],
    )
