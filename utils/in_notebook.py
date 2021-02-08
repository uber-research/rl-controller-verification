
def is_in_notebook(): 
    try: 
        get_ipython()
    except NameError:
        return False 
    return True
