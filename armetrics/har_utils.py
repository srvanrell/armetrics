import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from armetrics.models import Event, Segment
from armetrics.radar_chart import radar_factory


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
    events_labels = ["D", "F", "FM", "M", "C", "M'", "FM'", "F'", "I'"]
    for i, tag in enumerate(dic_summary_of_events):
        tags.append(tag)
        summary_of_events = dic_summary_of_events[tag]
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
    theta += np.pi / 12

    fig, axes = plt.subplots(figsize=(9, 7), nrows=1, ncols=1,
                             subplot_kw=dict(projection='radar'))

    half_axis = np.concatenate(([theta[0]-np.pi/12], theta[:6], [theta[5]+np.pi/12]))
    half_circle = [1] * 8

    axes.fill(half_axis, half_circle, facecolor="grey", alpha=0.25)  # Fill the polygon
    colors = ["C%d" % i for i in range(len(case_labels))]
    axes.set_rgrids([0.2, 0.4, 0.6, 0.8], [0.8, 0.6, 0.4, 0.2])
    axes.set_title(title, weight='bold', position=(0.5, 1.1),
                   horizontalalignment='center', verticalalignment='center')
    axes.set_ylim([0, 1])
    for d, color in zip(case_data, colors):
        axes.plot(theta, d, color=color)
        # axes.fill(theta, d, facecolor=color, alpha=0.25)  # Fill the polygon
    axes.set_varlabels(radial_labels)

    # Block- and frame-based legend
#    frame_patch = mpatches.Patch(facecolor='lightgrey', edgecolor='k', label='Frame-based metrics')
#    block_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Block-based metrics')
    frame_patch = mpatches.Patch(facecolor='lightgrey', edgecolor='k', label='Medidas por ventanas')
    block_patch = mpatches.Patch(facecolor='white', edgecolor='k', label='Medidas por bloques')
    first_legend = plt.legend(handles=[frame_patch, block_patch], ncol=1,
                              bbox_to_anchor=(-0.15, 0.0), loc=2, borderaxespad=0.0,
                              framealpha=0)
    plt.gca().add_artist(first_legend)

    axes.legend(case_labels, bbox_to_anchor=(1.15, 1.05),  # loc=(0.9, .95),
                loc=2, labelspacing=0.5, fontsize='small')

    plt.tight_layout(pad=3.5)
    plt.savefig(title + ".pdf")
    plt.savefig(title + ".png")
    plt.show()


def spider_df_summaries(summaries_by_activity, labels):
    for act in summaries_by_activity[0].mean().index.tolist():
        single_spider_df_summaries([s.get_group(act) for s in summaries_by_activity],
                                   labels, act)
        print_f1scores_df_summaries([s.get_group(act) for s in summaries_by_activity],
                                    labels, act)
        # violinplot_relative_errors_df_summaries([s.get_group(act) for s in summaries_by_activity],
        #                                         labels, act)
        violinplot_raw_errors_df_summaries([s.get_group(act) for s in summaries_by_activity],
                                           labels, act)


def single_spider_df_summaries(summaries, labels, title="Titulo"):
    case_data = []
    for summary, lab in zip(summaries, labels):
        summary_mean = summary.mean()
        # Saving data to plot
        case_data.append([summary_mean.frame_recall, summary_mean.frame_precision,
                          1 - summary_mean.f_rate, 1 - summary_mean.m_rate,
                          1 - summary_mean.d_rate, 1 - summary_mean.i_rate,
                          # 1 - summary_mean.u_rate, 1 - summary_mean.o_rate[act],
                          #
                          1 - summary_mean.ins_rate, 1 - summary_mean.del_rate,
                          1 - summary_mean.merge_rate, 1 - summary_mean.frag_rate,
                          summary_mean.event_precision, summary_mean.event_recall
                          ])

    spider_plot(title=title,
                radial_labels=[
                    # Frame-based metrics
                    # "Recall", "Precision", "1-Frag. rate", "1-Merg. rate", "1-Del. rate", "1-Ins. rate",
                    # Eje invertido
#                    "FNR", "FDR", "Frag.", "Merg.", "Del.", "Ins.",
                    "FNR", "FDR", "Frag.", "Comb.", "Elim.", "Ins.",
                    # Frame-based metrics
#                    "Ins.", "Del.", "Merg.", "Frag.", "FDR", "FNR"
                    "Ins.", "Elim.", "Comb.", "Frag.", "FDR", "FNR"
                ],
                case_data=case_data,
                case_labels=labels)


def print_f1scores_df_summaries(summaries, labels, act):
    print("Label".ljust(25) + "Frame-based f1score".ljust(25) + "Block-based f1score")
    for summary, lab in zip(summaries, labels):
        frame_str = "%0.3f (+-%0.3f)" % (summary.frame_f1score.mean(), summary.frame_f1score.std())
        event_str = "%0.3f (+-%0.3f)" % (summary.event_f1score.mean(), summary.event_f1score.std())
        print(lab.ljust(25)[:25] + frame_str.ljust(25) + event_str)


def violinplot_relative_errors_df_summaries(summaries, labels, act):
    plt.figure()
    pos = np.arange(len(labels)) + .5  # the bar centers on the y axis

    if len(labels) > 10:
        print("Be careful! I cannot plot more than 10 labels.")
    colors = ["C%d" % i for i in range(len(labels))]

    to_print = []
    for summary, lab, p in zip(summaries, labels, pos):
        time_errors = 100.0 * (summary.matching_time.as_matrix() - 1)
        to_print.append(" ".join(["%.2f" % i for i in time_errors]) +
                        "\t(%.2f)\t" % np.mean(time_errors) + lab)
        plt.violinplot(time_errors[np.isfinite(time_errors)], [p], points=50, vert=False, widths=0.5,
                       showmeans=True, showextrema=True, bw_method='silverman')

    plt.axvline(x=0, color="k", linestyle="dashed")
    plt.yticks(pos, labels)
    plt.gca().invert_yaxis()
    plt.xlabel('Time Prediction Error (%)')
    plt.show()

    print("Time prediction error per signal")
    for row in to_print:
        print(row)


def violinplot_raw_errors_df_summaries(summaries, labels, act):
    plt.figure()
    pos = np.arange(len(labels)) + .5  # the bar centers on the y axis

    if len(labels) > 10:
        print("Be careful! I cannot plot more than 10 labels.")
    colors = ["C%d" % i for i in range(len(labels))]

    to_print = []
    for summary, lab, p in zip(summaries, labels, pos):
        errors = summary.raw_time_error.as_matrix()
        to_print.append(" ".join(["%.2f" % i for i in errors]) +
                        "\t(%.2f)_mean\t" % np.mean(errors) +
                        "\t(%.2f)_median\t" % np.median(errors) + lab)
        plt.violinplot(errors[np.isfinite(errors)], [p], points=50, vert=False, widths=0.65,
                       showmeans=False, showmedians=True, showextrema=True, bw_method='silverman')

    plt.axvline(x=0, color="k", linestyle="dashed")
    plt.yticks(pos, labels)
    plt.gca().invert_yaxis()
    plt.xlabel('Time Prediction Error (frame)')
    plt.show()

    print("Time prediction error per signal")
    for row in to_print:
        print(row)
