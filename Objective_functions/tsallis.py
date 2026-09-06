import numpy as np
import math 
import common

#this function works on the single pile, and returns the Tsallis entropy for that pile
def class_tsallis_partial(class_slice, histogram, w, q):
    # class_slice = the list of shade values belonging to ONE pile
    # histogram = the full histogram, indexed by shade value
    # w = this pile's weight (from class_weight, already built by Otsu)
    # q = the fixed dial value (e.g. 0.8), same for the whole run

    total = 0
    for i in class_slice:
        if histogram[i] == 0:
            continue                     # skip empty bins
        p = histogram[i] / w             # re-normalize shade i within this pile
        total += math.pow(p, q)

    S_C = (1 - total) / (q - 1)
    return S_C

#this function takes the list of Tsallis entropies for each pile, and combines them into a single score
def combine_classes_tsallis(S_values, q):
    total_sum = sum(S_values)
    product = 1
    for S in S_values:
        product *= S                            # the extra step Kapur does NOT have —
#                                            # Tsallis entropy is "non-extensive",
    correction = (1 - q) * product           # so piles can't just be added together
    return total_sum + correction


def tsallis(thresholds, histogram, q):
    classes = common.split_into_classes(thresholds, histogram)   # reused from Otsu/Kapur

    S_values = []
    for C in classes:
        w = common.class_weight(C)                                # reused from Otsu/Kapur
        S_values.append(class_tsallis_partial(C, histogram, w, q))

    score = combine_classes_tsallis(S_values, q)
    return score       