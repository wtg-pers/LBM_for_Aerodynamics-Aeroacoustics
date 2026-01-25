


def setup_library(MODE):
    if MODE == 'cpu':
        import numpy as xp
        print(f"MODE: CPU (forced)")
        return xp
    
    else:
        try:
            import cupy as xp
            xp.cuda.Device(0).compute_capability
            print(f"MODE: GPU (forced) - GPU device: {xp.cuda.Device(). id}")
            return xp
        except (ImportError, Exception) as e:
            print(f" Warnning: Requested the GPU mode, but can't use it: {e}")
            print("Replace with Numpy.")
            import numpy as xp
            return xp


