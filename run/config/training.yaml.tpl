training:
 # chosen states among all_states = ['u','v','w','phi','theta','psi','p','q','r','q_p', 'q_q','q_r','e_p','e_q','e_r','thrust','cmd_phi','cmd_theta','cmd_psi','wg_x','wg_y','wg_z','wg_a','saturation',pwm_1]
 # q_* corresponds to the query to the corresponding axis
 # e_* corresponds to the error between the query and the corresponding angular rate
  used_states: {USED_STATES}
  step: 0
  env:
    # Supported values:
    # 0 (QuadcopterEnv)
    # 1 (QuadcopterEnv1)
    # 2 (QuadcopterEnv2) (default)
    # 3 (QuadcopterEnv3)
    # 5 (QuadcopterEnv5) (query generator)
    value: '0'
    params:
      # It allows to set the Stable Baseline `n_envs` for the Models
      # supporting it, otherwise it is ignored
      n_envs: 1

  logging:
    # The Proprietary Framework Section
    framework:
      log_filename:
        active: True
        value: log.txt
      console:
        active: True
      verbosity: 1
    # The Proprietary Framework Section
    stable_baselines:
      console:
        active: 1
      log_filename:
        active: 1
        value: log.txt
      verbosity: 1
    tensorflow:
      summary:
        active: False
      stable_baselines_native:
        active: False
      stable_baselines_tensors: 
        active: False
        list: ['critic_tf']
      tensorflow_tensors:
        active: False
        list: ['model/pi/Relu:0']
      events:
        active: False
        list: ['on_step']

  save_plots: no
  suffix: ''


  training_params: # Configures the Training Env and Quadcopter

    aero: # Aerodynamic effects
      enabled: {AERO}
      windgust:
        rest_range: [0.1, 0.2]
        h_range: [0.1, 0.2]
        magnitude_max: {MAGNITUDE_MAX}

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

      value: episodic
      switch: # Switching Logic
        active: False # Defines if the switching logic is active. WARNING: Still not supported
        target: continuous # Defines what is the target switching policy after the switch has happened
        time_perc: 0.9 # The percentage of training time after which the switch needs to happen

    quadcopter:                       # Quadcopter Setup
      saturation_motor: {SATURATION}  # Configures the saturation of the first motor
      reset_policy:                   # Quadcopter Reset Policy
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



  testing_params: # Configures the Testing Env and Quadcopter

    aero: # Aerodynamic effects
      enabled: {AERO}
      windgust:
        rest_range: [0.1, 0.2]
        h_range: [0.1, 0.2]
        magnitude_max: {MAGNITUDE_MAX}

    pid_rates: {PID_RATES}

    old_query: False

    query_generation: # Query Generation Logic. Value=[continuous, episodic]
      value: continuous

    quadcopter:                       # Quadcopter Setup
      saturation_motor: {SATURATION}  # Configures the saturation of the first motor
      reset_policy:                   # Quadcopter Reset Policy# Quadcopter Reset Policy
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



  model:
    # Currently suppored: ddpg, ppo, trpo, td3, sac. It is CASE SENSITIVE so be careful on this.
    name: {ALGO}
    # True: then the model identified by the path at `checkpoint` is
    # loaded and the rest of this substructure is ignored.
    # False: this is ignored and the rest of this substructure is used.
    load:
      value: False
      checkpoint_base_path: bazel_mount_input
      checkpoint_id: 1
    policy:
      value: mlp
      # Type can `standard` or `custom` and in the first case the rest is ignored,
      # in the second case it is possible to specify the number of layers
      # and how many neurons for them
      type:
        value: custom
        layers: {LAYER}

  save:
    tf_checkpoint: yes
    as_tf: no
    with_datetime: yes

  debug:
    active: True
    describe: False
    try_save:
      all_variables: False
      trainable_variables: False
      graph: False
      weights: False # NOTE: it does not work with PPO
    run_training_loop: False # DEPRECATED
    show_tensors:
      active: False
      list: ['critic_tf']

  n_steps: {N_STEPS}

  # Standard Values for Algo: DDPG=10000, PPO=1000, TRPO=100
  iterations_checkpoint: 100000

  # Activation settings
  # Suppored values for end of actor: tanh, cbv
  activation: relu
  activation_end_of_actor: {ACTIVATION_END_OF_ACTOR}

  action_noise:
    mu: 0
    name: OrnsteinUhlenbeck
    sigma: 0.5
  plots_dir: figures
  models_dir: models
  query_class: easy
  query_classes:
    easy:
      amplitude: [-0.2,0.2]
      duration: [0.5,0.8]
      max_diff: 0.3
      max_angle: 45
      distribution: uniform
    middle:
      amplitude: [-0.4,0.4]
      duration: [0.2,0.5]
      max_diff: 0.6
      max_angle: 45
      distribution: uniform
    hard:
      amplitude: [-0.6,0.6]
      duration: [0.1,0.2]
      max_diff: 0.9
      max_angle: 45
      distribution: uniform
