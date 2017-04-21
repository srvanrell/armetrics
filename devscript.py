import numpy as np

ground = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0])
output = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])


def frames2segments(y_true, y_pred):
    """
    Compute segment boundaries and compare y_true with y_pred.
    
    Segments are derived by comparing y_true with y_pred:
    any change in either y_pred or y_true marks a segment boundary.
    First-segment start-index is 0 and last-segment end-index is -1 (the pythonic way). 
      
    :param y_true: array_like
    ground truth
    
    :param y_pred: array_like
    prediction or classifier output
    
    :return: array_like, 2 columns
    start- and end-indexes of segments.
    first column corresponds to starts, and second column to ends
    """
    # start - and end - indexes
    # of
    # segments.That is,.

    y_true_breaks = np.flatnonzero(np.diff(y_true)) + 1  # detect changes in y_true
    y_pred_breaks = np.flatnonzero(np.diff(y_pred)) + 1  # detect changes in y_pred
    seg_breaks = np.union1d(y_true_breaks, y_pred_breaks)  # combine both changes
    seg_starts = np.append([0], seg_breaks)  # add 0 as the first start
    seg_ends = np.append(seg_breaks, [-1])  # append -1 as the last end

    return zip(seg_starts, seg_ends)


def label_segment(y_true_seg, y_pred_seg):
    """
    Compares y_true_seg with y_pred_seg and returns the corresponding label
    
    :param y_true_seg: true value of segment
    :param y_pred_seg: predicted value of segment
    :return: label that indicates True Positive, True Negative, False Positive or False Negative.
    Possible outcomes: "TP", "TN", "FP", or "FN".
    """
    # FIXME this should be beautified :/
    if y_true_seg and y_pred_seg:
        return "TP"
    elif y_true_seg and not y_pred_seg:
        return "FN"
    elif not y_true_seg and y_pred_seg:
        return "FP"
    else:
        return "TN"


for start, end in frames2segments(ground, output):
    print(start, end)
