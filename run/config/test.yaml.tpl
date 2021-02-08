test:
  # The Base Dir of the Training Results
  training_base_dir: /results/{EXP_DIR}

  # Identifies the Environment to be used for Testing. If type==standard
  # then the same env as training is used, if type==custom then the env id specified
  # in value is used
  env:
    type: standard
    value: 2

  suffix: .pkl #TBD
  save_plots: no # Possibly Deprecated

  # The number of episodes for each checkpoint to be tested with respect to
  n_episodes: 100

  # If yes then random query are generated otherwise one single query is used
  continuous: no

  # This section is related to the checkpoints to evaluate
  checkpoints:
    # Start Index for the Checkpoint Evaluation
    start_index: 1
    # End Index for the Checkpoint Evaluation
    end_index: 30
    # The step between the start_index and the end_index identifying
    # the checkpoints to test
    step: 1

  # Supported: old-continuous, old-episodic, target.
  # old-continuous: old continuous mode.
  # old-episodic: old episodic mode.
  # target: allows to define a set of target values"
  mode:
    type: old-continuous
    # Target Query. Still to be implemented.

  eval_properties_observer:
    is_active: {PROPERTIES_OBSERVER_ENABLED}

  testing_params:
    aero: # Aerodynamic effects
      enabled: {AERO}
      windgust:
        rest_range: [0.1, 0.2]
        h_range: [0.1, 0.2]
        magnitude_max: {MAGNITUDE_MAX}

    query_class: {QUERY_CLASS} # easy, middle or hard

    pid_rates: {PID_RATES} # None, pid_rates_crazyflie or pid_rates_better

    pid_thrust: {PID_THRUST} # pid_thrust_main or pid_thrust_original

    old_query: False

    query_generation: # Query Generation Logic. Value=[continuous, episodic]

      continuous: # Define the params of continuous query generation
        T_episode: 20.0 # Length of the full episode
        dt: 0.01 # Time Resolution
        dt_command: 0.03 # Time Resolution of commands

      episodic: # Define the params of continuous query generation
        T_episode: 1.0 # Length of the full episode
        dt: 0.01 # Time Resolution
        dt_command: 0.03 # Time Resolution of commands

      value: continuous
      switch: # Switching Logic
        active: False # Defines if the switching logic is active. WARNING: Still not supported
        target: continuous # Defines what is the target switching policy after the switch has happened
        time_perc: 0.9 # The percentage of training time after which the switch needs to happen

    quadcopter:                           # Quadcopter Setup
      saturation_motor: {SATURATION}      # Configures the saturation of the first motor
      reset_policy:                       # Quadcopter Reset Policy
        # pdf: [none, uniform]
        # - none: does not overrdide the default config
        # - uniform: overrides the default config
        #   - params: [min, max] of the distribution
        abs_z:
          pdf: uniform
          params: [0, 1.0]
        velocity_x:
          pdf: uniform
          params: [-1.0, 1.0]
        velocity_y:
          pdf: uniform
          params: [-1.0, 1.0]
        velocity_z:
          pdf: uniform
          params: [-1.0, 1.0]
        abs_roll:
          pdf: uniform
          params: [-0.3, 0.3]
        abs_pitch:
          pdf: uniform
          params: [-0.3,0.3]
        abs_yaw:
          pdf: uniform
          params: [-1.0, 1.0]
        rate_roll:
          pdf: none
          params: [-0.2, 0.2]
        rate_pitch:
          pdf: none
          params: [-0.2, 0.2]
        rate_yaw:
          pdf: none
          params: [-0.2, 0.2]


