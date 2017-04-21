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
    
    :return: tuple (3 columns), 
    (array_like) first column corresponds to starts, 
    (array_like) second column corresponds to ends,
    (list) third column corresponds to basic labels (TP, TN, FP, FN)
    """
    y_true_breaks = np.flatnonzero(np.diff(y_true)) + 1  # locate changes in y_true
    y_pred_breaks = np.flatnonzero(np.diff(y_pred)) + 1  # locate changes in y_pred
    seg_breaks = np.union1d(y_true_breaks, y_pred_breaks)  # define segment breaks
    seg_starts = np.append([0], seg_breaks)  # add 0 as the first start
    seg_ends = np.append(seg_breaks, [-1])  # append -1 as the last end
    # Compare segments at their first element to get corresponding labels
    seg_labels = [basic_segment_label(y_true[i], y_pred[i]) for i in seg_starts]

    return zip(seg_starts, seg_ends, seg_labels)


def basic_segment_label(y_true_seg, y_pred_seg):
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


def relabel_segments(segments):
    """
    Transform labels to:
    Insertion (Is)
    Merge (Ms)
    Overfill (Os). start (Oa), end (Oz)
    Deletion (Ds)
    Fragmenting (F)
    Underfill (Us), start (Ua), end (Uz)
    
    :param basic_labeled_segment: 
    :return: 
    """

    output = []
    #FIXME it consider that there are more than one segment
    # this may not be true

    #TODO add correct

    # First segment relabel
    aux = ''.join(segments[:2])
    if aux in ["FPTN", "FPFN"]:
        output.append("I")  # Insertion
    elif aux in ["FNTN", "FNFP"]:
        output.append("D")  # Deletion
    elif aux in ["FPTP"]:
        output.append("Oa")  # start Overfill
    elif aux in ["FNTP"]:
        output.append("Ua")  # start Underfill

    for i in range(1, len(segments)):
        aux = ''.join(segments[i-1:i+1])
        if aux in ["TPFPTP"]:
            output.append("M")  # Merge
        elif aux in ["TNFPTN", "FNFPTN", "TNFPFN", "FNFPFN"]:
            output.append("I")  # Insertion
        elif aux in ["TNFNTN", "FPFNTN", "TNFNFP", "FPFNFP"]:
            output.append("D")  # Deletion
        elif aux in ["TPFNTP"]:
            output.append("F")  # Fragmentation
        #TODO add over and under fill

    # TODO add last segment relabel

    # for i, seg in enumerate(segments):
    #
    #
    #
    #
    #     segment


    return output

for start, end, label in frames2segments(ground, output):
    print(start, end, label)
