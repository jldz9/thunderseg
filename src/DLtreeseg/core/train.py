#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model training related module in DLtreeseg
"""

import sys
from pathlib import Path

import torch, detectron2
from detectron2.model_zoo import get_config
from detectron2.config import LazyConfig, instantiate
from detectron2.config import LazyCall as L

