def environment_dimensions(env_model : str):
    if env_model == "Savinov_val3":
        return [-9, 6, -5, 4]
    elif env_model == "Savinov_val2":
        return [-5, 5, -5, 5]
    elif env_model == "Savinov_test7":
        return [-9, 6, -4, 4]
    elif env_model == "plane":
        return None
    elif "obstacle" in env_model:
        return None
    else:
        raise ValueError("No matching env_model found.")
