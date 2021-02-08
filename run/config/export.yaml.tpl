input:
  experiment_dir: /results/{EXP_DIR}
  stable_baseline_checkpoint_filename: quadcopter-final.pkl
  stable_baseline_checkpoint_subdir: models
output:
  console: no
  file: yes
  subdir: exports
debug:
  is_active: no
  show_tensors:
  - model/pi/fc0/kernel:0
  - model/pi/fc0/bias:0
