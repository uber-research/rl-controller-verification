import cloudpickle 

def get_stable_baseline_file_params(filename): 
    with open(filename, 'rb') as file_ : 
        data, params = cloudpickle.load(file_)
    return params 

