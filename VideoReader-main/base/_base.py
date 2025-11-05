"""
This python file defines the base class for modules
"""

class BaseModule:
    """
    Base class for all modules.
    """

    def __init__(self, name: str):
        """
        Initialize the module with a name.
        """
        self.name = name

    def run(self):
        """
        Run the module.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")