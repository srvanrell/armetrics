import numpy as np
import pandas as pd
import os

from armetrics.models import Event, Segment
from armetrics import scorer


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
    # Pad with zeros
    max_len = max(len(y_true), len(y_pred))
    y_true = np.pad(y_true, (0, max_len - len(y_true)), "constant")
    y_pred = np.pad(y_pred, (0, max_len - len(y_pred)), "constant")

    y_true_breaks = np.flatnonzero(np.diff(y_true)) + 1  # locate changes in y_true
    y_pred_breaks = np.flatnonzero(np.diff(y_pred)) + 1  # locate changes in y_pred
    seg_breaks = np.union1d(y_true_breaks, y_pred_breaks)  # define segment breaks
    seg_starts = np.append([0], seg_breaks)  # add 0 as the first start
    seg_ends = np.append(seg_breaks, [len(y_true)])  # append len(y_true) as the last end

    # Compare segments at their first element to get corresponding labels
    seg_basic_labels = [scorer.segment_basic_score(y_true[i], y_pred[i]) for i in seg_starts]
    segments = [Segment(start, end, label) for start, end, label in zip(seg_starts, seg_ends, seg_basic_labels)]
    if advanced_labels:
        segments = scorer.score_segments(segments)

    return segments


def binarize_frames(labeled_frames, class_to_keep):
    binarized = []
    for frame in labeled_frames:
        if frame == class_to_keep:
            binarized.append(1)
        else:
            binarized.append(0)
    return np.array(binarized, dtype=np.int8)


def binframes2events(bin_frames):
    breaks = np.flatnonzero(np.diff(bin_frames)) + 1  # locate changes in bin_frames
    starts = np.append([0], breaks)  # add 0 as the first start
    ends = np.append(breaks, [len(bin_frames)])  # append len(bin_frame) as the last end
    # Events are specified with one
    events = [[f_start, f_end] for f_start, f_end in zip(starts, ends) if bin_frames[f_start]]
    return events


def segments2frames(scored_segments):
    output = []
    for seg in scored_segments:
        output += [seg.label] * (seg.end - seg.start)
    return output


def events2frames(event_list, length=None):
    """
    Translate an event list into an array of binary frames.

    Event list comprising start and end indexes of events (must be positive).
     For example: [[3, 5], [8, 10]]

    Returns an np.array corresponding to frames.
     Frames that correspond to an event ar marked with 1.
     Frames that not correspond to an event ar marked with 0.

    :param length: (None by default)
     Extend the frame array to given length
    :param event_list:
    :return: frames:
    """
    frames = []
    for start_e, end_e in event_list:
        frames += [0] * (start_e - len(frames))
        frames += [1] * (end_e - start_e)
    if length:
        frames += [0] * (length - len(frames))
    return np.array(frames, dtype=np.int8)


def get_scores(y_true_bin, y_pred_bin):
    y_true_evs = binframes2events(y_true_bin)
    y_pred_evs = binframes2events(y_pred_bin)
    scored_segments = frames2segments(y_true_bin, y_pred_bin)
    scored_frames = segments2frames(scored_segments)
    scored_true_events, scored_pred_events = scorer.score_events(scored_segments, y_true_evs, y_pred_evs)

    return {"scored_true_events": scored_true_events,
            "scored_pred_events": scored_pred_events,
            "events_summary": scorer.events_summary(scored_true_events, scored_pred_events),
            "frames_summary": scorer.frames_summary(scored_frames)}


def get_sessions_scores(ground_filenames, prediction_filenames, loader_function, activities_of_interest, **kwarg):
    """
    :param ground_filenames: list of filenames for the ground truth
    :param prediction_filenames: list of filenames for the predictor
    :param activities_of_interest: list of labels of interest (among the ones within given files)
    :param loader_function: function to read files, it should return ?? FIXME
    Additions arguments are given to loader_function
    """

    # Ground sessions labels
    yground_by_session = [loader_function(filename, **kwarg) for filename in ground_filenames]
    # Prediction sessions labels
    ypred_by_session = [loader_function(filename, **kwarg) for filename in prediction_filenames]

    dfs_to_concat = []

    for ground_filename, pred_filename, ytest, ypred in zip(ground_filenames, prediction_filenames,
                                                            yground_by_session, ypred_by_session):
        for activity in activities_of_interest:
            ytest_bin = binarize_frames(ytest, activity)
            ypred_bin = binarize_frames(ypred, activity)
            scores_dic = get_scores(ytest_bin, ypred_bin)

            temp_merged = pd.DataFrame({**scores_dic["events_summary"],
                                        **scores_dic["frames_summary"],
                                        "activity":  activity,
                                        "ground_filename": os.path.basename(ground_filename),
                                        "prediction_filename": os.path.basename(pred_filename)
                                        }, index=[99])

            dfs_to_concat.append(temp_merged)

    df = pd.concat(dfs_to_concat, ignore_index=True)

    return df




