# 
# Copyright 2020-2022 SpireTech, Inc. All rights reserved.
# Description: 
# Generic Dockerfile for python application


# registry.access.redhat.com/ubi7/python-38:1-41
# Ref: https://catalog.redhat.com/software/containers/ubi7/python-38/5e8388a9bed8bd66f839abb3

ARG PYTH_BASE_IMAGE=registry.access.redhat.com/ubi7:7.9-516

FROM ${PYTH_BASE_IMAGE}

ENV PYTHON_VERSION=3.8 \
    PYTHON_SCL_VERSION=38 \
    PATH=$PATH:/opt/rh/rh-python38/root/usr/bin:$HOME/.local/bin/:/opt/rh/$NODEJS_SCL/root/usr/bin:/opt/rh/httpd24/root/usr/bin:/opt/rh/python38/root/usr/bin:/opt/rh/httpd24/root/usr/sbin:/opt/rh/rh-python38/root/usr/local/bin \
    LD_LIBRARY_PATH=/opt/rh/rh-python38/root/usr/lib64:/opt/rh/$NODEJS_SCL/root/usr/lib64:/opt/rh/httpd24/root/usr/lib64:/opt/rh/python38/root/usr/lib64 \
    LIBRARY_PATH=/opt/rh/httpd24/root/usr/lib64 \
    X_SCLS=rh-python38 \
    MANPATH=/opt/rh/rh-python38/root/usr/share/man:/opt/rh/python38/root/usr/share/man:/opt/rh/httpd24/root/usr/share/man:/opt/rh/$NODEJS_SCL/root/usr/share/man \
    VIRTUAL_ENV=/opt/app-root \
    APP_ROOT=/opt/app-root \
    PYTHONPATH=/opt/rh/$NODEJS_SCL/root/usr/lib/python2.7/site-packages \
    XDG_DATA_DIRS=/opt/rh/python38/root/usr/share:/opt/rh/rh-python38/root/usr/share:/usr/local/share:/usr/share \
    PKG_CONFIG_PATH=/opt/rh/python38/root/usr/lib64/pkgconfig:/opt/rh/httpd24/root/usr/lib64/pkgconfig:/opt/rh/rh-python38/root/usr/lib64/pkgconfig \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    CNB_STACK_ID=com.redhat.stacks.ubi7-python-38 \
    CNB_USER_ID=1001 \
    CNB_GROUP_ID=0 \
    PIP_NO_CACHE_DIR=off

LABEL "authur"="Spiretech.co" \
      "source-repo"="git@gitlab.com:ai.spiretech.co/predictivemoneymanagement/training/predictivemoneymanagement.training.model.git" \
      "copyright"="Copyright 2020-2022 SpireTech, Inc. All rights reserved."

WORKDIR /opt/app-root

RUN INSTALL_PKGS="rh-python38 rh-python38-python-devel rh-python38-python-setuptools rh-python38-python-pip nss_wrapper \
        httpd24 httpd24-httpd-devel httpd24-mod_ssl httpd24-mod_auth_kerb httpd24-mod_ldap \
        httpd24-mod_session atlas-devel gcc-gfortran libffi-devel libtool-ltdl enchant" && \
        yum install -y yum-utils && \        
        yum -y --setopt=tsflags=nodocs install $INSTALL_PKGS && \
        rpm -V $INSTALL_PKGS && \
        source scl_source enable rh-python38 && \
        python3 -m venv ${APP_ROOT} && \                
        python3 -m pip install --upgrade pip && \
        curl https://packages.microsoft.com/config/rhel/7/prod.repo > /etc/yum.repos.d/mssql-release.repo && \
        ACCEPT_EULA=Y yum remove unixODBC-utf16 unixODBC-utf16-devel && \
        ACCEPT_EULA=Y yum install -y gcc gcc-c++ && \
        ACCEPT_EULA=Y yum install -y msodbcsql17 && \
        ACCEPT_EULA=Y yum install -y unixODBC-devel && \
        # Create a user
        useradd -u 1001 -d ${APP_ROOT} -M python && \
        # Remove redhat-logos (httpd dependency) to keep image size smaller.
        rpm -e --nodeps redhat-logos && \
        yum -y clean all --enablerepo='*'


COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . .

RUN chown -R 1001:1001 ${APP_ROOT}

USER 1001

CMD [ "python3", "main.py"]

