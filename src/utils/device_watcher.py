import torch


class DeviceWatcher:

    device_list = ["cpu"]

    @classmethod
    def _initialize(cls):
        """Initialize device list by detecting available hardware accelerators."""
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            cuda_count = torch.cuda.device_count()
            for i in range(cuda_count):
                cls.device_list.append(f"cuda:{i}")

        # Check for MPS (Apple Silicon GPU)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            cls.device_list.append("mps")

    @classmethod
    def get_available_device(cls):
        return cls.device_list

# Only initalize here
DeviceWatcher._initialize()