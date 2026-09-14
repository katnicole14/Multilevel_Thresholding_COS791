import math 
import common

#this function works on the single pile, and returns the Tsallis entropy for that pile
def class_tsallis_partial(class_slice, histogram, w, q):
    # class_slice = the list of shade values belonging to ONE pile
    # histogram = the full histogram, indexed by shade value
    # w = this pile's weight (from class_weight, already built by Otsu)
    # q = the fixed dial value (e.g. 0.8), same for the whole run

    if w == 0:
        return 0.0

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
    terms = [1 + (1 - q) * S for S in S_values]
    product = 1
    for t in terms:
        product *= t
    return (product - 1) / (1 - q)


def tsallis(thresholds, histogram, q):
    classes = common.split_into_classes(thresholds, histogram)   # reused from Otsu/Kapur

    S_values = []
    for C in classes:
        w = common.class_weight(C, histogram)                     # reused from Otsu/Kapur
        S_values.append(class_tsallis_partial(C, histogram, w, q))

    score = combine_classes_tsallis(S_values, q)
    return score       