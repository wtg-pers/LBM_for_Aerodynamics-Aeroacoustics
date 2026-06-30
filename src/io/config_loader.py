import importlib.util
import os, sys


class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        spec = importlib.util.spec_from_file_location("user_config", self.config_path)
        if spec is None:
             raise ImportError(f"Cannot load config from {self.config_path}")
             
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if not hasattr(config_module, 'config'):
            raise AttributeError("Config file must have a 'config' dictionary.")
            
        return config_module.config