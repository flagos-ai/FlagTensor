%global debug_package %{nil}

# PyTorch and CUDA are supplied by the NVIDIA runtime rather than RPMs.
# Keep automatic requirements for distro libraries, Python, and TritonJIT.
%global __requires_exclude ^([(]python3([.][0-9]+)?dist[(](torch|triton)[)] .*[)]|python3([.][0-9]+)?dist[(](matplotlib|openpyxl|pyyaml)[)] .*|(libcuda[.]so[.]1|libtorch(_cpu|_cuda|_python)?[.]so|libc10(_cuda)?[.]so)[(][)][(]64bit[)])$

Name:           libflagtensor-nvidia
Version:        0.1.0
Release:        2%{?dist}
Summary:        FlagTensor C++ operator runtime (NVIDIA backend)

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagTensor
Source0:        flagtensor-%{version}.tar.gz

BuildRequires:  cmake >= 3.25
BuildRequires:  gcc-c++
BuildRequires:  libtriton-jit-nvidia-devel >= 0.1.0-3
BuildRequires:  ninja-build
BuildRequires:  patchelf
BuildRequires:  python3-devel
BuildRequires:  python3-pip
BuildRequires:  python3-rpm-macros
BuildRequires:  python3-setuptools
BuildRequires:  python3-wheel
Requires:       libtriton-jit-nvidia%{?_isa} >= 0.1.0-3

%description
Native FlagTensor operators backed by CUDA and the system Triton JIT runtime.
This package also contains the Triton kernel sources used at runtime.

%package devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       libtriton-jit-nvidia-devel%{?_isa} >= 0.1.0-3

%description devel
Headers and CMake package files for applications using FlagTensor operators.

%package -n python3-flagtensor
Summary:        FlagTensor Python operators for FlagOS
BuildArch:      noarch
Requires:       python3
Requires:       python3-matplotlib
Requires:       python3-openpyxl
Requires:       python3-pyyaml
Suggests:       python3-flagtensor-nvidia

%description -n python3-flagtensor
Python tensor operators and backend registration used by FlagOS workloads.
GPU-enabled PyTorch and Triton are supplied by the selected backend runtime.

%package -n python3-flagtensor-nvidia
Summary:        FlagTensor Python C++ extension (NVIDIA backend)
Requires:       %{name}%{?_isa} = %{version}-%{release}
Requires:       python3-flagtensor = %{version}-%{release}

%description -n python3-flagtensor-nvidia
Python bindings for the native NVIDIA FlagTensor operator runtime.

%prep
%autosetup -n flagtensor-%{version}

%build
PY3_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH=/usr/local/lib/python${PY3_VER}/site-packages:/usr/local/lib64/python${PY3_VER}/site-packages:$(python3 -c "import site; print(':'.join(site.getsitepackages()))")
export PATH=/usr/local/bin:$PATH
TORCH_CMAKE_PATH=$(python3 -c "import importlib.util, os; s=importlib.util.find_spec('torch'); print(os.path.join(os.path.dirname(s.origin), 'share', 'cmake'))")
%cmake -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="-Xcompiler -fPIE" \
    -DFLAGTENSOR_BACKEND=CUDA \
    -DFLAGTENSOR_BUILD_C_EXTENSIONS=ON \
    -DFLAGTENSOR_INSTALL=ON \
    -DFLAGTENSOR_PYTHON_INSTALL_DIR=%{python3_sitelib}/flagtensor \
    -DFLAGTENSOR_USE_EXTERNAL_TRITON_JIT=ON \
    -DTorch_ROOT="${TORCH_CMAKE_PATH}" \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON
%cmake_build

mkdir -p dist-rpm
python3 -m build --wheel --no-isolation \
    --outdir dist-rpm

%install
wheel=$(find dist-rpm -name 'flagtensor-*.whl' -print -quit)
python3 -m pip install --no-deps --no-compile --no-index \
    --target %{buildroot}%{python3_sitelib} "$wheel"
rm -f %{buildroot}%{python3_sitelib}/flagtensor-*.dist-info/RECORD
%cmake_install
find %{buildroot} -name '*.so*' -type f \
    -exec patchelf --remove-rpath {} \;

%check
test -f %{buildroot}%{_libdir}/libflagtensor.so.0
test -n "$(find %{buildroot}%{python3_sitelib}/flagtensor -name 'c_operators*.so' -print -quit)"
test -f %{buildroot}%{python3_sitelib}/flagtensor/runtime/backend/_nvidia/tune_configs.yaml
test -f %{buildroot}%{_libdir}/cmake/FlagTensor/FlagTensorConfig.cmake

%files
%license LICENSE
%doc README.md
%{_libdir}/libflagtensor.so.0*
%{_libdir}/flagtensor/triton_src/

%files devel
%{_includedir}/flagtensor/
%{_libdir}/libflagtensor.so
%{_libdir}/cmake/FlagTensor/

%files -n python3-flagtensor
%license LICENSE
%{python3_sitelib}/flagtensor/
%exclude %{python3_sitelib}/flagtensor/c_operators*.so
%{python3_sitelib}/flagtensor-*.dist-info/

%files -n python3-flagtensor-nvidia
%{python3_sitelib}/flagtensor/c_operators*.so

%changelog
* Fri Aug 07 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-2
- Build the NVIDIA native runtime and Python extension
- Link against the system libtriton-jit-nvidia package
- Keep backend-neutral Python files in a separate noarch package
- Preserve distro requirements while filtering NVIDIA/PyTorch SONAMEs
- Ship backend YAML configuration and accept distro dependency versions

* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM packaging
