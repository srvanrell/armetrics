import numpy as np
from armetrics.models import Event, Segment


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


def score_segments(basic_scored_segments):
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

    segments = [Segment(seg.start, seg.end) for seg in basic_scored_segments]

    # First segment relabel
    aux = ''.join(seg.label for seg in basic_scored_segments[:2])
    if basic_scored_segments[0].label in ["TP"]:
        segments[0].label = "C"  # Correct
    elif basic_scored_segments[0].label in ["TN"]:
        segments[0].label = ""  # Correct null
    elif aux in ["FPTN", "FPFN"]:
        segments[0].label = "I"  # Insertion
    elif aux in ["FNTN", "FNFP"]:
        segments[0].label = "D"  # Deletion
    elif aux in ["FPTP"]:
        segments[0].label = "Oa"  # starting Overfill
    elif aux in ["FNTP"]:
        segments[0].label = "Ua"  # starting Underfill

    # Middle segment relabel
    for i in range(1, len(basic_scored_segments) - 1):
        aux = ''.join(seg.label for seg in basic_scored_segments[i - 1:i + 2])
        if basic_scored_segments[i].label in ["TP"]:
            segments[i].label = "C"  # Correct
        elif basic_scored_segments[i].label in ["TN"]:
            segments[i].label = ""  # Correct null
        elif aux in ["TPFPTP"]:
            segments[i].label = "M"  # Merge
        elif aux in ["TPFNTP"]:
            segments[i].label = "F"  # Fragmentation
        elif aux in ["TNFPTN", "FNFPTN", "TNFPFN", "FNFPFN"]:
            segments[i].label = "I"  # Insertion
        elif aux in ["TNFNTN", "FPFNTN", "TNFNFP", "FPFNFP"]:
            segments[i].label = "D"  # Deletion
        elif aux in ["TNFPTP", "FNFPTP"]:
            segments[i].label = "Oa"  # starting Overfill
        elif aux in ["TPFPTN", "TPFPFN"]:
            segments[i].label = "Oz"  # ending Overfill
        elif aux in ["TNFNTP", "FPFNTP"]:
            segments[i].label = "Ua"  # starting Underfill
        elif aux in ["TPFNTN", "TPFNFP"]:
            segments[i].label = "Uz"  # ending Underfill

    if len(basic_scored_segments) > 1:
        # Last segment relabel
        aux = ''.join(seg.label for seg in basic_scored_segments[-2:])
        if basic_scored_segments[-1].label in ["TP"]:
            segments[-1].label = "C"  # Correct
        elif basic_scored_segments[-1].label in ["TN"]:
            segments[-1].label = ""  # Correct null
        elif aux in ["TNFP", "FNFP"]:
            segments[-1].label = "I"  # Insertion
        elif aux in ["TNFN", "FPFN"]:
            segments[-1].label = "D"  # Deletion
        elif aux in ["TPFP"]:
            segments[-1].label = "Oa"  # ending Overfill
        elif aux in ["TPFN"]:
            segments[-1].label = "Ua"  # ending Underfill

    return segments


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
    else:
        summary["event_recall"] = np.nan

        summary["frag_rate"] = np.nan
        summary["merge_rate"] = np.nan
        summary["del_rate"] = np.nan

    if summary["num_pred_events"] > 0:
        summary["event_precision"] = 1.0 * summary["C"] / summary["num_pred_events"]

        summary["ins_rate"] = 1.0 * sum(summary[l] for l in ["I'"]) / summary["num_pred_events"]
    else:
        summary["event_precision"] = np.nan
        summary["ins_rate"] = np.nan

    if summary["event_recall"] > 0 and summary["event_precision"] > 0:
        summary["event_f1score"] = 2 * summary["event_recall"] * summary["event_precision"] / (
            summary["event_recall"] + summary["event_precision"])
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
    if (summary["tp"] + summary["fp"]) > 0:
        summary["frame_precision"] = 1.0 * summary["tp"] / (summary["tp"] + summary["fp"])
    else:
        summary["frame_precision"] = np.nan
    if (summary["tp"] + summary["fn"]) > 0:
        summary["frame_recall"] = 1.0 * summary["tp"] / (summary["tp"] + summary["fn"])
    else:
        summary["frame_recall"] = np.nan

    if summary["frame_recall"] > 0 and summary["frame_precision"] > 0:
        summary["frame_f1score"] = 2 * summary["frame_recall"] * summary["frame_precision"] / (
            summary["frame_recall"] + summary["frame_precision"])
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
