import math 
  
def kapur_entropy(thresholds, probabilities, repair=False): #repair checks if the thresholds are valid and removes duplicates or out-of-bounds values
    L = len(probabilities)
    t = sorted(thresholds)

    if repair:
        cleaned = []

        for v in t:
            v = max(0, min(L - 2, v)) # clamp the threshold value to be within the valid range [0, L-2]

            if not cleaned or v > cleaned[-1]: # remove duplicates and ensure thresholds are in increasing order
                cleaned.append(v)

        t = cleaned # update the thresholds with the cleaned list

    # create a list of boundaries that includes the start (0), the thresholds incremented by 1, and the end (L)
    boundaries = [0] + [ti + 1 for ti in t] + [L] 

    #entropy begins at zero since no classes have been processed yet
    total_entropy = 0.0

    for i in range(len(boundaries) - 1): # go through each class defined by the boundaries
        start = boundaries[i]
        stop = boundaries[i + 1]

        class_mass = 0.0 # initialize the total probability mass for the current class

        for j in range(start, stop):
            class_mass += probabilities[j]

        if class_mass <= 0.0: # empty class, no pixels fall into this range, so the entropy is undefined
            # Avoid log(0) by returning negative infinity.
            return float("-inf")

        class_entropy = 0.0

        for j in range(start, stop):
            p = probabilities[j]

            if p > 0.0:
                cond = p / class_mass
                class_entropy -= cond * math.log(cond)

        total_entropy += class_entropy

    return total_entropy
