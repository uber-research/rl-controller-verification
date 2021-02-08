import architectures.convert 

def get_sherlock_format(model_desc, params): 
    """Returns a string representing the Sherlock Format of a Model 
        model_desc: JSON Description of a model like the one config_training['training']['model']
        params: Model Params
    """
    model = architectures.convert.name2model(model_desc=model_desc, params=params)
    return model.get_sherlock_header() + model.get_sherlock_content()


