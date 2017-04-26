import numpy as np

ground = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0])
output1 = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])


def frames2segments(y_true, y_pred, advanced_labels=True):
    """
    Compute segment boundaries and compare y_true with y_pred.
    
    Segments are derived by comparing y_true with y_pred:
    any change in either y_pred or y_true marks a segment boundary.
    First-segment start-index is 0 and last-segment end-index is len(y_true). 
      
    :param y_true: array_like
        ground truth
    
    :param y_pred: array_like
        prediction or classifier output

    :param advanced_labels: (Default True)
        Defines what kind of labels to return
    
    :return: tuple (3 columns), 
    (array_like) first column corresponds to starts, 
    (array_like) second column corresponds to ends,
    (list) third column corresponds to basic labels (TP, TN, FP, FN)
    or advanced labels (C, I, D, M, F, Oa, Oz, Ua, Uz)
    """
    y_true_breaks = np.flatnonzero(np.diff(y_true)) + 1  # locate changes in y_true
    y_pred_breaks = np.flatnonzero(np.diff(y_pred)) + 1  # locate changes in y_pred
    seg_breaks = np.union1d(y_true_breaks, y_pred_breaks)  # define segment breaks
    seg_starts = np.append([0], seg_breaks)  # add 0 as the first start
    seg_ends = np.append(seg_breaks, [len(y_true)])  # append len(y_true) as the last end
    # Compare segments at their first element to get corresponding labels
    seg_basic_labels = [segment_basic_score(y_true[i], y_pred[i]) for i in seg_starts]
    if advanced_labels:
        seg_labels = segment_score(seg_basic_labels)
    else:
        seg_labels = seg_basic_labels

    return zip(seg_starts, seg_ends, seg_labels)


def segment_basic_score(y_true_seg, y_pred_seg):
    """
    Compares y_true_seg with y_pred_seg and returns the corresponding label
    
    :param y_true_seg: true value of segment
    :param y_pred_seg: predicted value of segment
    :return: label that indicates True Positive, True Negative, False Positive or False Negative.
    Possible outcomes: "TP", "TN", "FP", or "FN".
    """
    true_vs_pred = {(True, True): "TP",
                    (True, False): "FN",
                    (False, True): "FP",
                    (False, False): "TN"}

    return true_vs_pred[(y_true_seg, y_pred_seg)]


def segment_score(basic_scored_segments):
    """
    Transform basic labels "TP", "TN", "FN", "FP" to:
    Correct (C)
    Correct Null ("")
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

    # First segment relabel
    aux = ''.join(basic_scored_segments[:2])
    if basic_scored_segments[0] in ["TP"]:
        output.append("C")  # Correct
    elif basic_scored_segments[0] in ["TN"]:
        output.append("")  # Correct null
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
        if basic_scored_segments[i] in ["TP"]:
            output.append("C")  # Correct
        elif basic_scored_segments[i] in ["TN"]:
            output.append("")  # Correct null
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
        if basic_scored_segments[-1] in ["TP"]:
            output.append("C")  # Correct
        elif basic_scored_segments[-1] in ["TN"]:
            output.append("")  # Correct null
        elif aux in ["TNFP", "FNFP"]:
            output.append("I")  # Insertion
        elif aux in ["TNFN", "FPFN"]:
            output.append("D")  # Deletion
        elif aux in ["TPFP"]:
            output.append("Oa")  # ending Overfill
        elif aux in ["TPFN"]:
            output.append("Ua")  # ending Underfill

    return output


def labeled_segments2labeled_frames(labeled_segments):
    output = []
    for start, end, label in labeled_segments:
        for i in range(start, end):
            output.append(label)
    return output

for start, end, label in frames2segments(ground, output1):
    print(start, end, label)


labeled = frames2segments(ground, output1)
a = labeled_segments2labeled_frames(labeled)
print(a, len(a), len(ground))
