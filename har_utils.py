import numpy as np
from models import Event, Segment
import matplotlib.pyplot as plt
from radar_chart import radar_factory


# TODO the shorter input should be pad with zeros
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
    return np.array(frames, dtype='int64')


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
               "M'": scored_pred_events.count("M")     # Total merging events
               }

    if normalize:
        # Normalized true events metrics
        for lab in ["C", "D", "F", "FM", "M"]:
            summary[lab+"_rate"] = summary[lab] / max(1, len(scored_true_events))
        # Normalized predicted events metrics
        for lab in ["C'", "I'", "F'", "FM'", "M'"]:
            summary[lab+"_rate"] = summary[lab] / max(1, len(scored_pred_events))

    return summary


def frames_summary(scored_frames, normalize=True):

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

    if normalize:
        # Normalized positives frame metrics
        total_positives = summary["tp"] + summary["d"] + summary["f"] + summary["ua"] + summary["uz"]
        for lab in ["tp", "d", "f", "ua", "uz"]:
            summary[lab+"_rate"] = summary[lab] / max(1, total_positives)
        # Normalized predicted events metrics
        total_negatives = summary["tn"] + summary["i"] + summary["m"] + summary["oa"] + summary["oz"]
        for lab in ["tn", "i", "m", "oa", "oz"]:
            summary[lab+"_rate"] = summary[lab] / max(1, total_negatives)

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


def plot_frame_pies(summary_of_frames):
    positive_labels = ["tp", "f", "d", "ua", "uz"]
    positive_labels = [lab for lab in positive_labels if summary_of_frames[lab] > 0]
    positive_counts = [summary_of_frames[lab] for lab in positive_labels]

    negative_labels = ["tn", "m", "i", "oa", "oz"]
    negative_labels = [lab for lab in negative_labels if summary_of_frames[lab] > 0]
    negative_counts = [summary_of_frames[lab] for lab in negative_labels]

    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.pie(positive_counts, labels=positive_labels, autopct='%1.1f%%',
            startangle=90, pctdistance=1.1, labeldistance=1.25)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.set_title('Positive frames')

    ax2.pie(negative_counts, labels=negative_labels, autopct='%1.1f%%',
            startangle=90, pctdistance=1.1, labeldistance=1.25)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax2.set_title('Negative frames')

    plt.tight_layout()
    plt.show()


def plot_event_bars(dic_summary_of_events):
    tags = []
    for i, tag in enumerate(dic_summary_of_events):
        tags.append(tag)
        summary_of_events = dic_summary_of_events[tag]
        events_labels = ["D", "F", "FM", "M", "C", "M'", "FM'", "F'", "I'"]
        events_counts = [summary_of_events[lab] for lab in events_labels]
        total_count = sum(events_counts)*1.0

        n_labels = len(events_counts)
        column = i
        bar_width = 0.2

        colors = plt.cm.Vega10(np.linspace(0, 1, n_labels))

        # Initialize the vertical-offset for the stacked bar chart.
        y_offset = 0.0

        # Plot bars
        p = [""] * n_labels
        for i_lab in range(n_labels):
            norm_count = events_counts[i_lab] / total_count
            p[i_lab] = plt.bar(column, norm_count, bar_width, bottom=y_offset, color=colors[i_lab])
            if events_counts[i_lab]:
                plt.text(column, y_offset + 0.5 * norm_count,
                         events_labels[i_lab] + " (%d)" % events_counts[i_lab])
            y_offset = y_offset + norm_count

    plt.xticks(range(len(tags)), tags)
    plt.legend(p, events_labels, loc="best", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

    plt.title("Event diagram per activity")

    plt.show()


def spider_plot(title, radial_labels, case_data, case_labels):
    """
    Minimum example of the inputs
    :param title: 'Titulo'
    :param radial_labels: ["medida 1", "medida 2", "medida 3"]
    :param case_data: [[0.1, 0.2, 0.5],
                       [0.3, 0.4, 0.6]]
    :param case_labels: ["Serie 1", "Serie 2"]
    :return:
    """
    n_radial = len(radial_labels)
    theta = radar_factory(n_radial, frame='polygon')

    fig, axes = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                             subplot_kw=dict(projection='radar'))

    colors = ['b', 'r', 'g', 'y', 'm']
    axes.set_rgrids([0.2, 0.4, 0.6, 0.8])
    axes.set_title(title, weight='bold', position=(0.5, 1.1),
                   horizontalalignment='center', verticalalignment='center')
    axes.set_ylim([0, 1])
    for d, color in zip(case_data, colors):
        axes.plot(theta, d, color=color)
        # axes.fill(theta, d, facecolor=color, alpha=0.25)  # Fill the polygon
    axes.set_varlabels(radial_labels)

    axes.legend(case_labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small')

    plt.show()


def spider_summaries(frame_summaries, event_summaries, labels):
    for act in frame_summaries[0].keys():
        case_data = []
        for fr_summary, ev_summary in zip(frame_summaries, event_summaries):
            # Frame based measures
            tp_frames = fr_summary[act]["tp"]
            fn_frames = sum(fr_summary[act][l] for l in ["f", "d", "ua", "uz"])
            fp_frames = sum(fr_summary[act][l] for l in ["i", "oa", "oz", "m"])

            recall_fr = 1.0 * tp_frames / (tp_frames + fn_frames)
            precision_fr = 1.0 * tp_frames / (tp_frames + fp_frames)
            total_time_accuracy = 0.5 * (tp_frames + fp_frames) / (tp_frames + fn_frames)

            # Frame based time measures
            positive_frames = tp_frames + fn_frames
            underfill_frames = (fr_summary[act]["ua"] + fr_summary[act]["uz"])
            overfill_frames = (fr_summary[act]["oa"] + fr_summary[act]["oz"])

            underfill_rate = 1.0 * underfill_frames / positive_frames
            overfill_rate = 1.0 * overfill_frames / positive_frames

            # Event based measures
            tp_events = ev_summary[act]["C"]
            ground_events = sum(ev_summary[act][l] for l in ["C", "F", "FM", "M", "D"])
            output_events = sum(ev_summary[act][l] for l in ["C'", "F'", "FM'", "M'", "I'"])

            recall_ev = 1.0 * tp_events / ground_events  # Esta bien la definicion?
            precision_ev = 1.0 * tp_events / output_events  # Esta bien la definicion?

            frag_rate = 1.0 * sum(ev_summary[act][l] for l in ["F", "FM"]) / ground_events
            merge_rate = 1.0 * sum(ev_summary[act][l] for l in ["M", "FM"]) / ground_events
            del_rate = 1.0 * sum(ev_summary[act][l] for l in ["D"]) / ground_events
            ins_rate = 1.0 * sum(ev_summary[act][l] for l in ["I'"]) / output_events

            # Saving data to plot
            case_data.append([recall_fr, precision_fr,
                              1 - underfill_rate, 1 - overfill_rate, total_time_accuracy,
                              recall_ev, precision_ev,
                              1 - frag_rate, 1 - merge_rate,
                              1 - del_rate, 1 - ins_rate])

        spider_plot(title=act,
                    radial_labels=["frame recall", "frame precision",
                                   "1 - underfill rate", "1 - overfill rate", "total time accuracy/2",
                                   "event recall", "event precision",
                                   "1-frag rate", "1-merge rate",
                                   "1-del rate", "1-ins rate"],
                    case_data=case_data,
                    case_labels=labels)
