from typing import Any

import yaml
import os

class MAMLParameters(object):
    
    def __init__(self, parameters):
        self._config = parameters
    
    def get(self, property_name: str) -> Any:
        """
        Return value associated with property_name in configuration

        :param property_name: name of parameter in configuration
        :return: value associated with property_name
        """
        return self._config.get(property_name, "Unknown Key")

    def get_property_description(self, property_name: str) -> str:
        """
        Return description of configuration property

        :param property_name: name of parameter to query for description
        :return: description of property in configuration
        """
        raise NotImplementedError # TODO: Is this worth doing? .yaml not particularly amenable 

    def set_property(self, property_name: str, property_value: Any, property_description: str=None) -> None:
        """
        Add to the configuration specification

        :param property_name: name of parameter to append to configuration
        :param property_value: value to set for property in configuration
        :param property_description (optional): description of property to add to configuration
        """
        if property_name in self._config:
            raise Exception("This field is already defined in the configuration. Use ammend_property method to override current entry")
        else:
            self._config[property_name] = property_value

    def ammend_property(self, property_name: str, property_value: Any, property_description: str=None) -> None:
        """
        Add to the configuration specification

        :param property_name: name of parameter to ammend in configuration
        :param property_value: value to ammend for property in configuration
        :param property_description (optional): description of property to add to configuration
        """
        if property_name not in self._config:
            raise Exception("This field is not defined in the configuration. Use set_property method to add this entry")
        else:
            self._config[property_name] = property_value

    def show_all_parameters(self) -> None:
        """
        Prints entire configuration
        """ 
        print(self._config)

    def save_configuration(self, save_path: str) -> None:
        """
        Saves copy of configuration to specified path. Particularly useful for keeping track of different experiment runs

        :param save_path: path to folder in which to save configuration
        """
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "config.yaml"), "w") as f:
            yaml.dump(self._config, f)

