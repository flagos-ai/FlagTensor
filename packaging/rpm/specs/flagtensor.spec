%global debug_package %{nil}

Name:           python3-flagtensor
Version:        0.1.0
Release:        1%{?dist}
Summary:        FlagTensor — tensor utilities for FlagOS

License:        Apache-2.0
URL:            https://github.com/flagos-ai/FlagTensor
Source0:        %{url}/archive/v%{version}/flagtensor-%{version}.tar.gz
BuildArch:      noarch
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools >= 60
BuildRequires:  python3-wheel
BuildRequires:  python3-pip
BuildRequires:  pyproject-rpm-macros

%description
Tensor manipulation utilities (reshape/permute/index helpers) used across the FlagOS operator libraries.

%prep
%autosetup -n flagtensor-%{version}

%build
%pyproject_wheel

%install
%pyproject_install
%pyproject_save_files flagtensor

%check
# Smoke find_spec test (no actual import) — verifies the built module
# lands at the expected sitelib path. Doesn't import the module so
# missing runtime deps (torch, triton, ...) don't trip the check;
# those are user-install-time concerns, not packaging concerns.
PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=%{buildroot}%{python3_sitelib} \
    python3 -c "import importlib.util; s = importlib.util.find_spec('flagtensor'); assert s and s.origin, 'flagtensor not findable'; print('OK: flagtensor at', s.origin)"

%files -f %{pyproject_files}
%license LICENSE*

%changelog
* Wed May 13 2026 FlagOS Contributors <contact@flagos.io> - 0.1.0-1
- Initial RPM packaging.
