import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from armetrics.models import Event, Segment
from armetrics.scorer import segment_basic_score, score_segments
from armetrics.plotter import spider_df_summaries

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
    seg_basic_labels = [segment_basic_score(y_true[i], y_pred[i]) for i in seg_starts]
    segments = [Segment(start, end, label) for start, end, label in zip(seg_starts, seg_ends, seg_basic_labels)]
    if advanced_labels:
        segments = score_segments(segments)

    return segments


def binarize_frames(labeled_frames, class_to_keep):
    binarized = []
    for frame in labeled_frames:
        if frame == class_to_keep:
            binarized.append(1)
        else:
            binarized.append(0)
    return np.array(binarized, dtype='int64')


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


def score_events(scored_segments, true_events, pred_events):
    # Create list of true and predicted events with empty labels
    scored_true_evs = [Event(start, end) for start, end in true_events]
    scored_pred_evs = [Event(start, end) for start, end in pred_events]

    # True events labeling, first pass (using labeled_segments)
    for true_ev in scored_true_evs:
        for seg in scored_segments:
            if true_ev.start <= seg.start <= true_ev.end:
                # In the first pass, D and F segments are assigned to true events
                if seg.label in ["D", "F"]:
                    true_ev.add_label(seg.label)

    # Pred events labeling, first pass (using labeled_segments)
    for pred_ev in scored_pred_evs:
        for seg in scored_segments:
            if pred_ev.start <= seg.start <= pred_ev.end:
                # In the first pass, I and M segments are assigned to pred events
                if seg.label in ["I", "M"]:
                    pred_ev.add_label(seg.label)

    # True events labeling, second pass (using labels of prediction)
    for true_ev in scored_true_evs:
        for pred_ev in scored_pred_evs:
            if pred_ev.overlap(true_ev):
                if pred_ev.label in ["M", "FM"]:
                    true_ev.add_label("M")

    # Pred events labeling, second pass (using labels of ground truth)
    for pred_ev in scored_pred_evs:
        for true_ev in scored_true_evs:
            if true_ev.overlap(pred_ev):
                if true_ev.label in ["F", "FM"]:
                    pred_ev.add_label("F")

    # If no label was assigned so far, then it is a correct detected event
    for true_ev in scored_true_evs:
        if true_ev.label == "":
            true_ev.add_label("C")

    for pred_ev in scored_pred_evs:
        if pred_ev.label == "":
            pred_ev.add_label("C")

    return scored_true_evs, scored_pred_evs


def events_summary(scored_true_events, scored_pred_events, normalize=True):
    scored_true_events = [e.label for e in scored_true_events]
    scored_pred_events = [e.label for e in scored_pred_events]

    summary = {"C": scored_true_events.count("C"),     # Total correct events
               # ground truth
               "D": scored_true_events.count("D"),     # Total deleted events
               "F": scored_true_events.count("F"),     # Total fragmented events
               "FM": scored_true_events.count("FM"),   # Total fragmented and merged events
               "M": scored_true_events.count("M"),     # Total merged events
               # predicted output
               "C'": scored_pred_events.count("C"),    # Total correct events (equivalent to C if not normalized)
               "I'": scored_pred_events.count("I"),    # Total inserted events
               "F'": scored_pred_events.count("F"),    # Total fragmenting events
               "FM'": scored_pred_events.count("FM"),  # Total fragmenting and merging events
               "M'": scored_pred_events.count("M"),     # Total merging events
               "num_true_events": len(scored_true_events),
               "num_pred_events": len(scored_pred_events)
               }

    if normalize:
        # Normalized true events metrics
        for lab in ["C", "D", "F", "FM", "M"]:
            if summary["num_true_events"] > 0:
                summary[lab+"_rate"] = summary[lab] / max(1, summary["num_true_events"])
            else:
                summary[lab + "_rate"] = np.nan
        # Normalized predicted events metrics
        for lab in ["C'", "I'", "F'", "FM'", "M'"]:
            if summary["num_pred_events"] > 0:
                summary[lab+"_rate"] = summary[lab] / max(1, summary["num_pred_events"])
            else:
                summary[lab + "_rate"] = np.nan

    if summary["num_true_events"] > 0:
        summary["event_recall"] = 1.0 * summary["C"] / summary["num_true_events"]

        summary["frag_rate"] = 1.0 * sum(summary[l] for l in ["F", "FM"]) / summary["num_true_events"]
        summary["merge_rate"] = 1.0 * sum(summary[l] for l in ["M", "FM"]) / summary["num_true_events"]
        summary["del_rate"] = 1.0 * sum(summary[l] for l in ["D"]) / summary["num_true_events"]
    elif summary["num_pred_events"] == 0:
        # TODO REVIEW if there were no positives events in the ground truth then recall is set to 1.0
        # nothing can be say about fragmentation, merging or deletion so I will set them to 0
        summary["event_recall"] = 1.0  # np.nan

        summary["frag_rate"] = 0.0  # np.nan
        summary["merge_rate"] = 0.0  # np.nan
        summary["del_rate"] = 0.0  # np.nan
    else:
        # TODO REVIEW if there were no positives events in the ground truth then recall is set to 1.0
        # nothing can be say about fragmentation, merging or deletion so I will set them to 0
        summary["event_recall"] = np.nan

        summary["frag_rate"] = np.nan
        summary["merge_rate"] = np.nan
        summary["del_rate"] = np.nan

    if summary["num_pred_events"] > 0:
        summary["event_precision"] = 1.0 * summary["C"] / summary["num_pred_events"]

        summary["ins_rate"] = 1.0 * sum(summary[l] for l in ["I'"]) / summary["num_pred_events"]
    elif summary["num_true_events"] == 0:
        # TODO REVIEW if there were no predicted events in the output sequence then precision is set to 1.0
        # nothing can be say about insertion so I will set it to 0
        summary["event_precision"] = 1.0  # np.nan
        summary["ins_rate"] = 0.0  # np.nan
    else:
        # TODO REVIEW if there were no predicted events in the output sequence then precision is set to 1.0
        # nothing can be say about insertion so I will set it to 0
        summary["event_precision"] = np.nan
        summary["ins_rate"] = np.nan

    if summary["event_recall"] > 0 and summary["event_precision"] > 0:
        summary["event_f1score"] = 2 * summary["event_recall"] * summary["event_precision"] / (
            summary["event_recall"] + summary["event_precision"])
    elif summary["event_recall"] == 0 and summary["event_precision"] == 0:
        summary["event_f1score"] = 0.0  # np.nan
    else:
        summary["event_f1score"] = np.nan

    return summary


def frames_summary(scored_frames, normalize=True):
    # TODO add docstring

    summary = {"tp": scored_frames.count("C"),    # Total correct frames (true positives)
               "tn": scored_frames.count(""),     # Total correct frames (true negatives)
               "d": scored_frames.count("D"),     # Total deleted frames
               "f": scored_frames.count("F"),     # Total fragmented frames
               "i": scored_frames.count("I"),     # Total inserted frames
               "m": scored_frames.count("M"),     # Total merged frames
               "ua": scored_frames.count("Ua"),   # Total starting underfill frames
               "uz": scored_frames.count("Uz"),   # Total ending underfill frames
               "oa": scored_frames.count("Oa"),   # Total starting overfill frames
               "oz": scored_frames.count("Oz"),   # Total ending overfill frames
               }

    summary["u"] = summary["ua"] + summary["uz"]  # Total underfill frames
    summary["o"] = summary["oa"] + summary["oz"]  # Total overfill frames
    summary["ground_positives"] = sum(summary[lab] for lab in ["tp", "f", "d", "ua", "uz"])
    summary["ground_negatives"] = sum(summary[lab] for lab in ["tn", "m", "i", "oa", "oz"])
    summary["output_positives"] = sum(summary[lab] for lab in ["tp", "m", "i", "oa", "oz"])
    summary["output_negatives"] = sum(summary[lab] for lab in ["tn", "f", "d", "ua", "uz"])

    summary["fp"] = summary["output_positives"] - summary["tp"]
    summary["fn"] = summary["output_negatives"] - summary["tn"]

    # Given a confusion matrix:
    #
    #           predicted
    #           (+)   (-)
    #            ---------
    #       (+) | TP | FN |
    # actual     ---------
    #       (-) | FP | TN |
    #            ---------
    #
    # we know that:
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    #
    # Lets consider the cases where the denominator is zero:
    #
    # TP + FN = 0: means that there were no positive cases in the input data
    # TP + FP = 0: means that all instances were predicted as negative
    #
    # If TP = 0 (as in both cases), recall is 1, since the method has discovered all of none true positives;
    # precision is 0 if there is any FP and 1 otherwise

    if (summary["tp"] + summary["fp"]) > 0:
        summary["frame_precision"] = 1.0 * summary["tp"] / (summary["tp"] + summary["fp"])
    elif summary["fn"] == 0:
        # NOT SURE if there were no positives in the predicted sequence then precision is
        # not defined, here it is set to 1.
        summary["frame_precision"] = 1.0  # np.nan
    else:
        summary["frame_precision"] = np.nan

    if (summary["tp"] + summary["fn"]) > 0:
        summary["frame_recall"] = 1.0 * summary["tp"] / (summary["tp"] + summary["fn"])
    elif summary["fp"] == 0:
        # REVIEW if there were no positives in the ground truth then recall is set to 1.0
        # if FP=0 then ti should be 10. Im not sure it is right when FP != 0
        summary["frame_recall"] = 1.0  # np.nan
    else:
        summary["frame_recall"] = np.nan

    if summary["frame_recall"] > 0 and summary["frame_precision"] > 0:
        summary["frame_f1score"] = 2 * summary["frame_recall"] * summary["frame_precision"] / (
            summary["frame_recall"] + summary["frame_precision"])
    elif summary["frame_recall"] == 0 and summary["frame_precision"] == 0:
        summary["frame_f1score"] = 0.0  # np.nan
    else:
        summary["frame_f1score"] = np.nan

    if normalize:
        # Normalized positives frame metrics
        for lab in ["tp", "d", "f", "ua", "uz", "u", "fp"]:
            if summary["ground_positives"] > 0:
                summary[lab+"_rate"] = summary[lab] / max(1, summary["ground_positives"])
            else:
                summary[lab + "_rate"] = np.nan
        # Normalized predicted events metrics
        for lab in ["tn", "i", "m", "oa", "oz", "o", "fn"]:
            if summary["ground_negatives"] > 0:
                summary[lab+"_rate"] = summary[lab] / max(1, summary["ground_negatives"])
            else:
                summary[lab + "_rate"] = np.nan

    # FIXME Chequear que este funcionando bien cuando no hay etiquetas positivas en la referencia
    summary["raw_time_error"] = summary["output_positives"] - summary["ground_positives"]
    # if summary["tp"] + summary["fn"] > 0:
    summary["matching_time"] = summary["output_positives"] / max(1, summary["tp"] + summary["fn"])  # Matching time
    # else:
    #     summary["matching_time"] = np.nan

    return summary


def get_scores(y_true_bin, y_pred_bin):
    y_true_evs = binframes2events(y_true_bin)
    y_pred_evs = binframes2events(y_pred_bin)
    scored_segments = frames2segments(y_true_bin, y_pred_bin)
    scored_frames = segments2frames(scored_segments)
    scored_true_events, scored_pred_events = score_events(scored_segments, y_true_evs, y_pred_evs)

    return {"scored_true_events": scored_true_events,
            "scored_pred_events": scored_pred_events,
            "events_summary": events_summary(scored_true_events, scored_pred_events),
            "frames_summary": frames_summary(scored_frames)}


# TODO test this function in an experiment
def get_sessions_scores(ytest_by_session, ypred_by_session, classes_of_interest):
    """ (NOT IMPLEMENTED) average_mode should control if any average should be done (macro, micro, samples, ...).
    Open discussion involves if averaging should be done across sessions and/or across activities.
    """
    df = pd.DataFrame()

    for sid, (ytest, ypred) in enumerate(zip(ytest_by_session, ypred_by_session)):
        for act in classes_of_interest:
            ytest_bin = binarize_frames(ytest, act)
            ypred_bin = binarize_frames(ypred, act)
            scores_dic = get_scores(ytest_bin, ypred_bin)

            temp_df = pd.DataFrame(scores_dic["events_summary"], index=[sid])
            temp2_df = pd.DataFrame(scores_dic["frames_summary"], index=[sid])

            temp_merged = pd.concat([temp_df, temp2_df], axis=1)
            temp_merged["activity"] = act

            df = pd.concat([df, temp_merged])

    df.reset_index(inplace=True)
    df.rename(columns={"index": "session"}, inplace=True)

    return df
