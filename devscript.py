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
    seg_basic_labels = [segment_basic_score(y_true[i], y_pred[i]) for i in seg_starts]
    seg_labels = segment_score(seg_basic_labels)

    return zip(seg_starts, seg_ends, seg_labels)


def segment_basic_score(y_true_seg, y_pred_seg):
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


def segment_score(basic_scored_segments):
    """
    Transform basic labels "TP", "TN", "FN", "FP" to:
    Correct (C)
    Insertion (I)
    Merge (M)
    Overfill (O). starting (Oa), ending (Oz)
    Deletion (D)
    Fragmenting (F)
    Underfill (U), starting (Ua), ending (Uz)
    
    :param basic_scored_segments:
    List of basic scores assigned to segments
    :return: 
    List of advanced scores assigned to segments
    """

    output = []
    # FIXME it consider that there are more than one segment
    # this may not be true

    # First segment relabel
    aux = ''.join(basic_scored_segments[:2])
    if basic_scored_segments[0] in ["TP", "TN"]:
        output.append("C")  # Correct
    elif aux in ["FPTN", "FPFN"]:
        output.append("I")  # Insertion
    elif aux in ["FNTN", "FNFP"]:
        output.append("D")  # Deletion
    elif aux in ["FPTP"]:
        output.append("Oa")  # starting Overfill
    elif aux in ["FNTP"]:
        output.append("Ua")  # starting Underfill

    # Middle segment relabel
    for i in range(1, len(basic_scored_segments)-1):
        aux = ''.join(basic_scored_segments[i-1:i+2])
        if basic_scored_segments[i] in ["TP", "TN"]:
            output.append("C")  # Correct
        elif aux in ["TPFPTP"]:
            output.append("M")  # Merge
        elif aux in ["TPFNTP"]:
            output.append("F")  # Fragmentation
        elif aux in ["TNFPTN", "FNFPTN", "TNFPFN", "FNFPFN"]:
            output.append("I")  # Insertion
        elif aux in ["TNFNTN", "FPFNTN", "TNFNFP", "FPFNFP"]:
            output.append("D")  # Deletion
        elif aux in ["TNFPTP", "FNFPTP"]:
            output.append("Oa")  # starting Overfill
        elif aux in ["TPFPTN", "TPFPFN"]:
            output.append("Oz")  # ending Overfill
        elif aux in ["TNFNTP", "FPFNTP"]:
            output.append("Ua")  # starting Underfill
        elif aux in ["TPFNTN", "TPFNFP"]:
            output.append("Uz")  # ending Underfill

    if len(basic_scored_segments) > 1:
        # Last segment relabel
        aux = ''.join(basic_scored_segments[-2:])
        if basic_scored_segments[-1] in ["TP", "TN"]:
            output.append("C")  # Correct
        elif aux in ["TNFP", "FNFP"]:
            output.append("I")  # Insertion
        elif aux in ["TNFN", "FPFN"]:
            output.append("D")  # Deletion
        elif aux in ["TPFP"]:
            output.append("Oa")  # ending Overfill
        elif aux in ["TPFN"]:
            output.append("Ua")  # ending Underfill

    return output

for start, end, label in frames2segments(ground, output):
    print(start, end, label)
