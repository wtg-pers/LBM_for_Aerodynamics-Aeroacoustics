import json
import os

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)

        return config
    
    def get_simulation_params(self):
        return self.config.get('simulation', {})
    
    def get_boundary_config(self, location):
        boundaries = self.config.get('boundaries', {})
        return boundaries.get(location, None)
    
    def get_all_boundaries(self):
        return self.config.get('boundaries', {})
    
    def get_internal_geometry(self):
        return self.config.get('internal_geometry', {})
    
    def get_output_config(self):
        return self.config.get('output', {})
    
    def parse_location(self, location_expr, domain_shape):
        """
        위치 표현식 파싱
        
        Parameters:
        -----------
        location_expr : str or int, 예: "Nx-1", 0, "Ny//2"
        domain_shape : tuple, (Nx, Ny) or (Nx, Ny, Nz)
        
        Returns:
        --------
        int : parsed location
        """
        if isinstance(location_expr, int):
            return location_expr
        
        # 문자열 표현식 평가
        Nx, Ny = domain_shape[0], domain_shape[1]
        Nz = domain_shape[2] if len(domain_shape) > 2 else None
        
        try:
            # eval 사용 (주의: 신뢰할 수 있는 입력만)
            location = eval(location_expr, {"Nx": Nx, "Ny": Ny, "Nz": Nz})
            return int(location)
        except Exception as e:
            raise ValueError(f"Cannot parse location expression '{location_expr}': {e}")