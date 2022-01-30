from typing import List, Tuple
import string

class Module():

    _id : string
    moduleType : string
    position : Tuple[int, int]
    endpoints : List[Tuple[int, int]]

    def __init__(self, moduleType, position, endpoints) -> None:

        self.moduleType = moduleType
        self.position = position
        self.endpoints = endpoints

        self._id = hash(moduleType + str(position))


class Connection():

    removedConnection : bool
    modules : Tuple[Module, Module]

    def __init__(self, module_A, module_B) -> None:
        
        self.modules = (module_A, module_B)