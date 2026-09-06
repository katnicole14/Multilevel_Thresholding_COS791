def split_into_classes(thresholds, histogram):
    classes = []
    start = 0
    for t in thresholds:
        classes.append(list(range(start, t + 1)))
        start = t + 1
    classes.append(list(range(start, len(histogram))))
    return classes

def class_weight(C):
#     return sum(histogram[i] for i in C)   #suppose to connect with histogram function
    return 1   # hardcoded placeholder until real histogram is wired in

