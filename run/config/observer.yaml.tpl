prop_obs:

  aggregate:
    episodes: {AGGREGATE_EPISODES}
    checkpoints: {AGGREGATE_CHECKPOINTS}
    experiments: {AGGREGATE_EXPERIMENTS}
    checkpoint_id: {AGGREGATE_CHECKPOINT_ID}
    testing_base_dir: {TESTING_BASE_DIR}

  load_signals:
    t_is_hex: True
    tlimit: 5000

  po_params:
    # lookahead for query stability in secs.
    query_lh: 0.01
    # settling time in secs.
    tset: 0.25
    # stability time in secs.
    tstab: 0.5
    # stability margin for the query (absolute).
    query_diam: 0.05
    # 10% offset error percentage relative to stepsize.
    offset_pct: 0.1
    # 10% overshoot error percentage relative to stepsize.
    overshoot_pct: 0.1
    # perctentage for the time to reach prop (absolute).
    rising_time_pct: 0.05
