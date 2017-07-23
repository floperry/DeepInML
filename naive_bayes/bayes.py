#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np

class NaiveBayesClassifier(object):

	def train(self, dataset, classes):
		''' train naive bayes classifier
		'''

		# sorted by classes
		sub_datasets = defaultdict(lambda: [])
		cls_cnt = defaultdict(lambda: 0)

		for doc_vect, cls in zip(dataset, classes):
			sub_datasets[cls].append(doc_vect)
			cls_cnt[cls] += 1

		# compute classes prob
		cls_probs = {k: v/len(classes) for k, v in cls_cnt.items()}

		# compute conditional prob
		cond_probs = {}
		dataset = np.array(dataset)

		for cls, sub_dataset in sub_datasets.items():
			sub_dataset = np.array(sub_dataset)
			cond_prob_vect = np.log((np.sum(sub_dataset, axis=0) + 1)/(np.sum(dataset) + 2))
			cond_probs[cls] = cond_prob_vect

		return cond_probs, cls_probs

	def classify(self, doc_vect, cond_probs, cls_probs):
		''' use naive bayes classifier to classify
		'''

		pred_probs = {}

		for cls, cls_prob in cls_probs.items():
			cond_prob_vect = cond_probs[cls]
			pred_probs[cls] = np.sum(cond_prob_vect*doc_vect) + np.log(cls_prob)

		return max(pred_probs, key=pred_probs.get)