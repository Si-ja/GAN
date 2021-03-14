def activate_device(device="cuda"):
    """
    This function helps activate the device which will be doing the processing
    for the neural network. It is advised to use cuda with Nvidia GPUs if
    such ones exists on the hosts computer. If it does not, a CPU can always
    be used as a default alternative.
    
    This function prepares an object that will represent the device on which
    all the calculation operation are working.

    Parameters
    ----------
    device : TYPE, string
        DESCRIPTION. The default is "cuda".
        You can indicate what device you want to put your data and operations
        onto. The choise is pretty much is either "cuda" or "cpu". If your 
        computer does not support "cuda" or simple doesn't have it, even if
        you indicate the device to be "cuda" it will safe check it and will
        put data on the cpu.

    Returns
    -------
    device (object) - an object to which data will be stored.

    """   
    import torch
    
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    
    elif device == "cpu":
        device = "cpu"
        return device
    
    else:
        device = "cpu"
        return device