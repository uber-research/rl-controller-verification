# because of the ugly way configuration are managed in the project,
# we are forced to extend rule `doe_config` with some stuff we prefer
# to hide here...
load("@brezel//rules/doe:config_factory.bzl", atcp_doe_config="doe_config")
load("@brezel//rules/doe:config_factory.bzl", atcp_doe_config_param="doe_config_param")
load("@brezel//rules/doe:config_factory.bzl", "DoeConfigInfo")


def _result_path(config_name, config_file):
    prefix = config_name + '_training_'
    path = config_file.basename.replace(prefix, '')
    path = path.replace('.yaml', '')
    path = path.replace('_', '/')
    return path

def _test_config_from_training(config, ctx):
    test = ctx.actions.declare_file(config.basename.replace('_training_', '_test_'))
    train_dir = _result_path(ctx.label.name, config)
    subst = dict({"{TRAINING_DIR}": train_dir}, **ctx.attr.testing_substitutions)
    ctx.actions.expand_template(
        template = ctx.file.testing_template,
        output = test,
        substitutions = subst,
    )
    return test

def _obsv_config_from_training(config, ctx):
    obs = ctx.actions.declare_file(config.basename.replace('_training_', '_observer_'))
    test_dir = _result_path(ctx.label.name, config).replace('/training/', '/testing/')
    if ctx.attr.nominal:
        test_dir = 'NOMINAL/' + test_dir
    subst = dict({"{TESTING_DIR}": test_dir}, **ctx.attr.observer_substitutions)
    ctx.actions.expand_template(
        template = ctx.file.observer_template,
        output = obs,
        substitutions = subst,
    )
    return obs

def _doe_config_bundle_impl(ctx):
    outs = []
    tests = []
    obsvs = []
    for config in ctx.files.training_configs:
        out = ctx.actions.declare_file('main_'+config.basename)
        test = _test_config_from_training(config, ctx)
        obsv = _obsv_config_from_training(config, ctx)
        ctx.actions.expand_template(
            template = ctx.file.template,
            output = out,
            substitutions = {
                "{CONFIG_TRAINING}": config.short_path,
                "{CONFIG_EXPORT}": "",
                "{CONFIG_TEST}": test.short_path,
                "{CONFIG_OBSERVER}": obsv.short_path,
                "{OUTPUT}": _result_path(ctx.label.name, config),
            },
        )
        outs.append(out)
        tests.append(test)
        obsvs.append(obsv)

    exps = []
    naming = {}
    for cfg in ctx.attr.training_configs:
        naming = dict(cfg[DoeConfigInfo].naming, **naming)
        exps = exps + cfg[DoeConfigInfo].exps

    return [
        DefaultInfo(files = depset(outs+tests+obsvs)),
        DoeConfigInfo(
            files = outs,
            exps = exps,
            naming = naming,
            prefix = 'main_'+ctx.label.name+'_training',
        ),
    ]

_doe_config_bundle = rule(
    implementation = _doe_config_bundle_impl,
    attrs = {
        "training_configs": attr.label_list(allow_files=True),
        "testing_template": attr.label(allow_single_file=True),
        "testing_substitutions": attr.string_dict(),
        "observer_template": attr.label(allow_single_file=True),
        "observer_substitutions": attr.string_dict(),
        "template": attr.label(allow_single_file=True, default="//run/config:config.yaml.tpl"),
        "nominal": attr.bool(),
    },
)

def doe_config(name, visibility=None, **kwargs):
    native.genrule(
        name = "_{}_training_template".format(name),
        outs = ["_{}_training_template.yaml.tpl".format(name)],
        srcs = ["//run/config:training.yaml.tpl"],
        cmd = "sed 's/with_datetime: yes/with_datetime: no/' $< > $@",
    )
    native.genrule(
        name = "_{}_testing_template".format(name),
        outs = ["_{}_testing_template.yaml.tpl".format(name)],
        srcs = ["//run/config:test.yaml.tpl"],
        cmd = "sed '/^test:/a \  training_gcs_path: {GCS_BUCKET}/training/{TRAINING_DIR}' $< > $@",
    )
    native.genrule(
        name = "_{}_observer_template".format(name),
        outs = ["_{}_observer_template.yaml.tpl".format(name)],
        srcs = ["//run/config:observer.yaml.tpl"],
        cmd = "sed '/^prop_obs:/a \  testing_gcs_path: {GCS_BUCKET}/testing/{TESTING_DIR}' $< > $@",
    )
    atcp_doe_config(
        name = "{}_training".format(name),
        template = "_{}_training_template.yaml.tpl".format(name),
        visibility = visibility,
        **kwargs
    )
    _doe_config_bundle(
        name = name,
        training_configs = [":{}_training".format(name)],
        testing_template = "_{}_testing_template.yaml.tpl".format(name),
        testing_substitutions = kwargs["substitutions"] if "substitutions" in kwargs else {},
        observer_template = "_{}_observer_template.yaml.tpl".format(name),
        observer_substitutions = kwargs["substitutions"] if "substitutions" in kwargs else {},
        visibility = visibility,
    )
    native.filegroup(
        name = "{}.files".format(name),
        srcs = [":{}".format(name)] + ["{}_training".format(name)],
        visibility = visibility,
    )

doe_config_param = atcp_doe_config_param
