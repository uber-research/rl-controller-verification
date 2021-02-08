from gym.envs.registration import register

register(
    id='quadcopter-v0',
    entry_point='gym_quadcopter.envs:QuadcopterEnv',
)

register(
    id='quadcopter-v1',
    entry_point='gym_quadcopter.envs:QuadcopterEnv1',
)

register(
    id='quadcopter-v2',
    entry_point='gym_quadcopter.envs:QuadcopterEnv2',
)

register(
    id='quadcopter-v3',
    entry_point='gym_quadcopter.envs:QuadcopterEnv3',
)

register(
    id='quadcopter-v3334',
    entry_point='gym_quadcopter.envs:QuadcopterEnv_3_3_3_4',
)

register(
    id='quadcopter-v4',
    entry_point='gym_quadcopter.envs:QuadcopterEnv4',
)


register(
    id='quadcopter-v5',
    entry_point='gym_quadcopter.envs:QuadcopterEnv5',
)
