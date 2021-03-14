def date_label():
    """
    Generate a label with which an image can be saved.
    It default to the current time the user is operating
    under, with minutes and seconds considered.

    Returns
    -------
    time_label (string) - a string identifier showing when an image
    has been generated.
    """
    from datetime import datetime
    
    now = datetime.now()
    time_label = now.strftime("%d_%m_%Y %H_%M_%S")
    return time_label