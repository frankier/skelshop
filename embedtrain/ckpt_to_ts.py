#!/usr/bin/env python

import sys

import torch

from embedtrain.litmod import MetGcnLit

print("Usage: ckpt_to_ts.py kps inckpt outts")

kps = int(sys.argv[1])

model = MetGcnLit.load_from_checkpoint(sys.argv[2])
traced = torch.jit.trace(model.gcn.eval(), torch.rand(1, 2, 1, kps, 1))
traced.save(sys.argv[3])
