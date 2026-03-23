import torch

from . import backend_utils
from . import get_vendor_info, gen_torch_device_object


class DeviceDetector(object):
    _instance = None

    def __new__(cls, *args, **kargs):
        if cls._instance is None:
            cls._instance = super(DeviceDetector, cls).__new__(cls)
        return cls._instance

    def __init__(self, vendor_name=None):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.info = self.get_vendor(vendor_name)
            self.vendor_name = self.info.vendor_name
            self.name = self.info.device_name
            self.dispatch_key = (
                self.name.upper()
                if self.info.dispatch_key is None
                else self.info.dispatch_key
            )
            self.device_count = gen_torch_device_object(self.vendor_name).device_count()
            self.support_fp64 = True
            self.support_bf16 = True
            self.support_int64 = True

    def get_vendor(self, vendor_name=None):
        if vendor_name is not None:
            return get_vendor_info(vendor_name)
        if torch.cuda.is_available():
            return get_vendor_info('nvidia')
        raise RuntimeError('No supported device found for flagtensor runtime')
