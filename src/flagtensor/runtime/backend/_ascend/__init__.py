# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name='ascend',
    device_name='npu',
    device_query_cmd='npu-smi info',
)
# Ascend910 family. torch.npu.get_device_properties().major returns None on
# current torch_npu builds, so arch specialization is disabled by default.
# Map major version strings here once torch_npu exposes real capability info.
ARCH_MAP = {}
CUSTOMIZED_UNUSED_OPS = ()

__all__ = ['*']
