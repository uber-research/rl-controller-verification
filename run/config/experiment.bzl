load("@brezel//rules/doe:config_factory.bzl", "config_filename")

def _config_file_impl(ctx):
    # declare output file
    if ctx.outputs.filename:
        out = ctx.outputs.filename
    else:
        filename = config_filename(ctx.label.name, ctx.attr.values) + ctx.attr.extension
        out = ctx.actions.declare_file(filename)

    # generate file by expanding template
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = out,
        substitutions = {"{%s}" % k.upper(): v for k,v in ctx.attr.values.items()},
    )

    return [DefaultInfo(files = depset([out]))]


config_file = rule(
    implementation = _config_file_impl,
    attrs = {
        "template": attr.label(mandatory=True, allow_single_file=True),
        "values": attr.string_dict(mandatory=True),
        "extension": attr.string(default=".yaml"),
        "filename": attr.output(),
    },
)


def _main_config_file_impl(ctx):
    out = ctx.outputs.out if ctx.outputs.out else ctx.actions.declare_file("config.yaml")
    ctx.actions.expand_template(
        template = ctx.file.template,
        output = out,
        substitutions = {
            "{CONFIG_TRAINING}": ctx.file.config_training.short_path,
            "{CONFIG_EXPORT}": ctx.file.config_export.short_path,
            "{CONFIG_TEST}": ctx.file.config_test.short_path,
            "{CONFIG_OBSERVER}": ctx.file.config_observer.short_path,
            "{OUTPUT}": "experiment",
        },
    )
    return [DefaultInfo(files = depset([out]))]


main_config_file = rule(
    implementation = _main_config_file_impl,
    attrs = {
        "template": attr.label(allow_single_file=True, default="//run/config:config.yaml.tpl"),
        "config_training": attr.label(allow_single_file=True, mandatory=True),
        "config_export": attr.label(allow_single_file=True, mandatory=True),
        "config_test": attr.label(allow_single_file=True, mandatory=True),
        "config_observer": attr.label(allow_single_file=True, mandatory=True),
        "deps": attr.label_list(),
        "out": attr.output(),
    }
)


def experiment_config(name, **kwargs):
    for tpl, vals in kwargs.items():
        config_file(
            name = "{}_{}".format(name, tpl),
            template = "//run/config:{}.yaml.tpl".format(tpl),
            values = vals,
            filename = "{}_{}.yaml".format(name, tpl),
        )

    main_config_file(
        name = "{}_yaml".format(name),
        config_training = ":{}_training".format(name),
        config_export = ":{}_export".format(name),
        config_test = ":{}_test".format(name),
        config_observer = ":{}_observer".format(name),
        deps = [":{}_{}".format(name, tpl) for tpl in kwargs.keys()],
        out = "{}.yaml".format(name),
    )

    native.filegroup(
        name = name,
        srcs = ["{}_yaml".format(name)] +[
            ":{}_{}".format(name, tpl) for tpl in kwargs.keys()
        ]
    )
