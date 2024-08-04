#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

import os
import sys
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
#METEOR_JAR = 'meteor-1.5.jar'
# print METEOR_JAR

class Meteor:

    def __init__(self):
        pass

    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []
        for i in imgIds:
            assert(len(res[i]) == 1)
            gt = gts[i]
            hy = res[i][0]
            gt = [word_tokenize(g) for g in gt]
            hy = word_tokenize(hy)

            score = round(meteor_score(gt, hy), 4)
            scores.append(score)
        #print('{}\n'.format(eval_line))
        #self.meteor_p.stdin.write('{}\n'.format(eval_line))
        #print(self.meteor_p.stdout.readline().strip())

        #for i in range(0,len(imgIds)):
        #    scores.append(float(self.meteor_p.stdout.readline().strip()))
        #score = float(self.meteor_p.stdout.readline().strip())
        #self.lock.release()

        return sum(scores)/len(scores), scores

    def method(self):
        return "METEOR"

