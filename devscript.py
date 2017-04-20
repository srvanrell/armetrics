import numpy as np

ground = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0])
output = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0])


def frames2segments(y_true, y_pred):
    """
    Compute start- and end-indexes of segments.
    
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

    y_true_breaks = np.flatnonzero(np.diff(y_true))
    y_pred_breaks = np.flatnonzero(np.diff(y_pred))
    seg_breaks = np.union1d(y_true_breaks, y_pred_breaks) + 1
    seg_starts = np.append([0], seg_breaks)
    seg_ends = np.append(seg_breaks, [-1])

    return zip(seg_starts, seg_ends)

for start, end in frames2segments(ground, output):
    print(start, end)
