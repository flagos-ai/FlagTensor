from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name='thead', device_name='cuda', device_query_cmd='ppu-smi'
)
ARCH_MAP = {}
CUSTOMIZED_UNUSED_OPS = ()

__all__ = ['*']
