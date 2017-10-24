"""WrightTools decorators and metaclasses."""


# --- import --------------------------------------------------------------------------------------


import os


# ---- decorators ---------------------------------------------------------------------------------


def group_singleton(cls):
    def getinstance(*args, **kwargs):
        # extract
        filepath = args[0] if len(args) > 0 else kwargs.get('filepath', None)
        parent = args[1] if len(args) > 1 else kwargs.get('parent', None)
        name = args[2] if len(args) > 2 else kwargs.get('name', 'data')
        # parse
        if filepath is None:
            instance = cls(*args, **kwargs)
            cls.instances[instance.fullpath] = instance
            filepath = instance.filepath
        if parent is None:
            parent = ''
            name = '/'
        fullname = filepath + '::' + parent + name
        # create and/or return
        if not fullname in cls.instances.keys():
            cls.instances[fullname] = cls(*args, **kwargs)
        return cls.instances[fullname]
    return getinstance
