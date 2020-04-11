# coding: utf-8


import inspect


class VisualSelectionTaskInstance(object):
    
    def __init__(self, 
        task_name,     # text
        instance_id,   # text
        image_path,    # text
        tokens,        # ndarray with dtype int
        n_tokens,      # ndarray with dtype int
        object_bboxes, # ndarray with dtype float32
        object_optional_info,   # None or ndarray with dtype float32 and 768 dim
        n_objects,       # ndarray with dtype int
        ground_truth_id, # ndarray with dtype int
    ):
        # set arguments as attributes
        local_dict = locals()
        for a in inspect.getfullargspec(self.__init__).args:
            (a == 'self') or setattr(self, a, local_dict[a])
        del local_dict


class DatasetReaderBase(object):
    
    control_tokens = {'main':[], 'sub':[]}
    
    @classmethod
    def register(cls, router):
        router[cls.task_name] = cls
    
    @classmethod
    def instance_to_text(cls, inst):
        """Eeturns a space-splitted text given an instance."""
        raise NotImplementedError()
    
    @classmethod
    def count_tokens(cls, dataset_spec, n_tokens):
        """Return a dict whose key and value are token and its frequency."""
        raise NotImplementedError()
    
    @classmethod
    def compile_dataset(cls, dataset_sepc, textifier):
        """Returns a list of VisualSelectionTaskInstance."""
        raise NotImplementedError()
