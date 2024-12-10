# Registry instance
from typing import Any
from collections import defaultdict


class RetriverFactory:
    retriver_methods: dict = defaultdict(dict)

    def register_retriever_method(
            self,
            type: str,
            method_name: str,
            method_func=None  # can be any classes or functions
    ):
        if self.has_retriver_method(type, method_name):
            return

        self.retriver_methods[type][method_name] = method_func

    def has_retriver_method(self, type: str, key: str) -> Any:
        return (type in self.retriver_methods) and (key in self.retriver_methods[type])

    def get_method(self, type, key) -> Any:
        return self.retriver_methods.get(type).get(key)
    
    
RETRIEVER_REGISTY = RetriverFactory()


def register_retriever_method(type, method_name):
    """ Register a new chunking method
    
    This is a decorator that can be used to register a new chunking method.
    The method will be stored in the self.methods dictionary.
    
    Parameters
    ----------
    method_name: str
        The name of the chunking method.
    """

    def decorator(func):
        """ Register a new chunking method """
        RETRIEVER_REGISTY.register_retriever_method(type, method_name, func)

    return decorator



def get_retriever_operator(type, method_name):
    retriever_operator = RETRIEVER_REGISTY.get_method(type, method_name)
    return retriever_operator


