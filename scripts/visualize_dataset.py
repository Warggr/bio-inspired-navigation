# coding: utf-8
from system.controller.reachability_estimator.training.H5Dataset import H5Dataset
dataset = H5Dataset("system/controller/reachability_estimator/training/data/reachability/dataset+spikings+distances.hd5")
get_ipython().system(' ls system/controller/reachability_estimator/training/')
dataset = H5Dataset("system/controller/reachability_estimator/data/reachability/dataset+spikings+distances.hd5")
next(dataset)
next([1, 2, 3])
next(iter(dataset))
