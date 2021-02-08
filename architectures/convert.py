
import architectures.a2c.ddpg.actor as actor

def name2model(model_desc, params):
    model_name = model_desc.get_model_name()
    if model_desc.get_actor_feature_extractor_name() != 'mlp':
        raise NotImplementedError(f"Exporting Policy Type {model_desc.get_actor_feature_extractor_name()} is unsupported at the moment")
    # default configuration for DDPG
    a2c_params = {
        'params': params,
        'layers_fc_base_name': "model/pi/fc",
        'layers_final_name': "model/pi/pi",
        'kernel_name': "kernel",
        'bias_name': "bias"
    }
    if model_name == 'ppo' or model_name == 'trpo':
        a2c_params = {
            'params': params,
            'layers_fc_base_name': "model/pi_fc",
            'layers_final_name': "model/pi",
            'kernel_name': "w",
            'bias_name': "b"
        }
    elif model_name == 'td3' or model_name =='sac':
        a2c_params['layers_final_name'] = "model/pi/dense"
    return actor.A2C_MLP_Actor_Base(**a2c_params)
