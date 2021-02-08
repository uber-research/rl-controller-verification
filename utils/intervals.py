import numpy as np


matrix = np.array(
  [[1. , -0.5, -0.5, -1.],
  [1. , -0.5,  0.5,  1.],
  [1. ,  0.5,  0.5, -1.],
  [1., 0.5, -0.5, 1. ]]
)

def get_pwm_intervals(low, high):
  pwm = np.zeros((4, 2), dtype=float)
  pwm[0, 0] = low[0] - 0.5 * (high[1] + high[2]) - high[3]
  pwm[0, 1] = high[0] - 0.5 * (low[1] + low[2]) - low[3]

  pwm[1, 0] = low[0] - 0.5 * (high[1] - low[2]) + low[3]
  pwm[1, 1] = high[0] - 0.5 * (low[1] - high[2]) + high[3]

  pwm[2, 0] = low[0] + 0.5 * (low[1] + low[2]) - high[3]
  pwm[2, 1] = high[0] + 0.5 * (high[1] + high[2]) - low[3]

  pwm[3, 0] = low[0] + 0.5 * (low[1] - high[2]) + low[3]
  pwm[3, 1] = high[0] + 0.5 * (high[1] - low[2]) + high[3]

  return pwm
