# -*- coding: utf-8 -*-
"""
Copyright 2020-2022 Spiretech.co, Inc. All rights reserved.
Licenses: LICENSE.md
Description: Utility file for creating a logger object.
Reference: https://spiretech.atlassian.net/wiki/spaces/PA/pages/592183297/HE-FUC-015+Authorization

# Reference:
- https://developers.google.com/protocol-buffers/docs/proto3
- 
  

# Change History
[23042022/MFB/SIN-416] - Authentication and Authorization
"""


import logging
import sys
from environs               import Env

env = Env()

# logger = logging

logging.basicConfig(level=env.log_level('PINO_LOG_LEVEL',logging.INFO),
                    format=env.str('PINO_LOG_FORMAT', '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s'),
                    datefmt="%d/%b/%Y %H:%M:%S",
                    stream=sys.stdout)