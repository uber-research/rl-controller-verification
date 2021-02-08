
class A2C_MLP_Actor_Base: 
    """ A2C DDPG specific with Multi Layer Perceptron Actor 
    """

    def __init__(self, params, layers_fc_base_name, layers_final_name, kernel_name="kernel", bias_name="bias", max_layers=100): 
        self.max_layers = max_layers
        self.params = params
        self.layers_fc_base_name = layers_fc_base_name
        self.layer_final_name = layers_final_name
        self.kernel_name = kernel_name
        self.bias_name = bias_name
        first_kernel, _ = self._layer_name2kernel_and_bias(layer_name=self._layer_index2layer_name(current_index=0, final_layer_index=self.max_layers))
        last_kernel, _ = self._layer_name2kernel_and_bias(layer_name=self._layer_index2layer_name(current_index=self.max_layers, final_layer_index=self.max_layers))
        self.input_dim = self.params[first_kernel].shape[0]
        self.output_dim = self.params[last_kernel].shape[1]
        self.layers_size = self._params2layers_size(params=params, layers_fc_base_name=layers_fc_base_name) 
        self.num_layers = len(self.layers_size)

    def _params2layers_size(self, params, layers_fc_base_name, max_layers=100): 
        res = []
        for i in range(max_layers): 
            try: 
                k, _ = self._layer_name2kernel_and_bias(layer_name=self._layer_index2layer_name(current_index=i, final_layer_index=max_layers))
                res.append(params[k].shape[1])
            except: 
                return res 
        return res

    def get_sherlock_header(self): 
        res = ""
        res += f"{self.input_dim}\n"
        res += f"{self.output_dim}\n"
        res += f"{self.num_layers}\n"
        for x in self.layers_size: res += f"{x}\n"
        return res 

    def _layer_index2layer_name(self, current_index, final_layer_index): 
        # Fully Connected Layers 
        if current_index < final_layer_index: return f"{self.layers_fc_base_name}{current_index}"
        # Final Layer 
        else: return f"{self.layer_final_name}"

    def _layer_name2kernel_and_bias(self, layer_name): 
        #name_layer = self._layer_index2layer_name(current_index=l, final_layer_index=self.num_layers)
        #name_kernel = f"{name_layer}/{self.kernel_name}:0"
        #name_bias = f"{name_layer}/{self.bias_name}:0"
        return f"{layer_name}/{self.kernel_name}:0", f"{layer_name}/{self.bias_name}:0"

    def get_sherlock_content(self, debug_file=False): 
        print(f"{self.params.keys()}")
        #print(f"{tensors_kernel.shape}")
        #print(f"{tensors_bias.shape}")
        res = ""

        # Considering also final layer
        for l in range(self.num_layers+1): 
            name_kernel, name_bias = self._layer_name2kernel_and_bias(layer_name=self._layer_index2layer_name(current_index=l, final_layer_index=self.num_layers))

            #name_kernel = f"{self.layers_name[l]}/{self.kernel_name}:0"
            #name_bias = f"{self.layers_name[l]}/{self.bias_name}:0"

            #name_kernel = f'model/{self.module_name}/fc{l}/kernel:0'
            #name_bias = f'model/{self.module_name}/fc{l}/bias:0'
            w = self.params[name_kernel]
            b = self.params[name_bias]
            print(f"{name_kernel} = {w.shape}")
            print(f"{name_bias} = {b.shape}")
            #w = tensors_kernel[l]
            #b = tensors_bias[l]
            #print(f"w = {w}")
            #rint(f"b = {b}")
            for i in range(b.shape[0]): 
                for j in range(w.shape[0]): 
                    if debug_file: res += f"w({j}, {i}) = "
                    res += f"{w[j][i]}\n"
                if debug_file: res += f"b({i}) = "
                res += f"{b[i]}\n"
        return res 


# Stndard A2C DDPG MLP Actor 
# See https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ddpg/policies.py#L134
#A2C_DDPG_MLP_Actor_Standard = A2C_MLP_Actor_Base(layers_fc_base_name="model/pi/fc", layers_final_name="model/pi/pi", kernel_name="kernel", bias_name="bias")
#A2C_DDPG_MLP_Actor_Standard = A2C_MLP_Actor_Base(input_dim=14, output_dim=3, layers_size=[64, 64], layers_fc_base_name="model/pi/fc", layers_final_name="model/pi/pi", layers_name=["model/pi/fc0", "model/pi/fc1", "model/pi/pi"], kernel_name="kernel", bias_name="bias")

# See https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/policies.py
#A2C_Common_MLP_Actor_Standard = A2C_MLP_Actor_Base(layers_fc_base_name="model/pi/fc", layers_final_name="model/pi/pi", kernel_name="w", bias_name="b")
#A2C_Common_MLP_Actor_Standard = A2C_MLP_Actor_Base(input_dim=14, output_dim=3, layers_size=[64, 64], layers_fc_base_name="model/pi/fc", layers_final_name="model/pi/pi", layers_name=["model/pi_fc0", "model/pi_fc1", "model/pi"], kernel_name="w", bias_name="b")
