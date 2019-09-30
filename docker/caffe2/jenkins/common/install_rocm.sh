#!/bin/bash

set -ex

install_ubuntu() {
    apt-get update
    apt-get install -y wget
    apt-get install -y libopenblas-dev

    # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
    apt-get install libc++1
    apt-get install libc++abi1

    apt-get install -y dpkg-dev

    mkdir -p /usr/repos
    cd /usr/repos/
    JOB=166
    wget --recursive --no-parent http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/compute-rocm-dkms-no-npi-hipclang-$JOB/
    dpkg-scanpackages . /dev/null | gzip -9c > Packages.gz
    cd -

    # Add rocm repository
    echo "deb file:/usr/repos ./" > /etc/apt/sources.list.d/rocm.list
    apt-get update --allow-insecure-repositories


    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
                   rocm-dev \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   rocblas \
                   hipsparse \
                   rocrand \
                   hipcub \
                   rocthrust \
                   rccl \
                   rocprofiler-dev
}

install_centos() {

  yum update -y
  yum install -y wget
  yum install -y openblas-devel

  yum install -y epel-release
  yum install -y dkms kernel-headers-`uname -r` kernel-devel-`uname -r`

  echo "[ROCm]" > /etc/yum.repos.d/rocm.repo
  echo "name=ROCm" >> /etc/yum.repos.d/rocm.repo
  echo "baseurl=http://repo.radeon.com/rocm/yum/rpm/" >> /etc/yum.repos.d/rocm.repo
  echo "enabled=1" >> /etc/yum.repos.d/rocm.repo
  echo "gpgcheck=0" >> /etc/yum.repos.d/rocm.repo

  yum update -y

  yum install -y \
                   rocm-dev \
                   rocm-utils \
                   rocfft \
                   miopen-hip \
                   rocblas \
                   hipsparse \
                   rocrand \
                   rccl \
                   hipcub \
                   rocthrust \
                   rocprofiler-dev \
                   roctracer-dev
}
 
# Install Python packages depending on the base OS
if [ -f /etc/lsb-release ]; then
  install_ubuntu
elif [ -f /etc/os-release ]; then
  install_centos
else
  echo "Unable to determine OS..."
  exit 1
fi
