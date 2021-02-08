

def f_fwtf_get_feed_dict(model): 
    feed_dict = {model.actions: model.stats_sample['actions']}

    for placeholder in [model.action_train_ph, model.action_target, model.action_adapt_noise, model.action_noise_ph]:
        if placeholder is not None:
            feed_dict[placeholder] = model.stats_sample['actions']

    for placeholder in [model.obs_train, model.obs_target, model.obs_adapt_noise, model.obs_noise]:
        if placeholder is not None:
            feed_dict[placeholder] = model.stats_sample['obs']

    return feed_dict




