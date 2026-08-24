from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="mthreads",
    device_name="musa",
    device_query_cmd="mthreads-gmi",
    dispatch_key="PrivateUse1",
)
ARCH_MAP = {}
CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
