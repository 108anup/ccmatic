def flatten(l) -> list:
    ret = []
    if(isinstance(l, list)):
        for item in l:
            ret.extend(flatten(item))
        return ret
    else:
        return [l]