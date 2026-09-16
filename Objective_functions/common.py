def split_into_classes(thresholds, histogram):
    thresholds = sorted(int(round(t)) for t in thresholds)   # sort + force to int
    classes = []
    start = 0
    for t in thresholds:
        classes.append(list(range(start, t + 1)))
        start = t + 1
    classes.append(list(range(start, len(histogram))))
    return classes

def class_weight(C, histogram):
    return sum(histogram[i] for i in C)

