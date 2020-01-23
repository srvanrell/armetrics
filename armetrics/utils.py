import numpy as np
import pandas as pd
import os

from armetrics.models import Segment
from armetrics import scorer
from armetrics import plotter


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


def get_sessions_scores(ground_filenames, prediction_filenames, loader_function, activities_of_interest,
                        starts_ends=None, **kwarg):
    """
    :param ground_filenames: list of filenames for the ground truth
    :param prediction_filenames: list of filenames for the predictor
    :param activities_of_interest: list of labels of interest (among the ones within given files)
    :param loader_function: function to read files, it should return a pandas Series
    (see load_txt_labels() for an example of load)
    :param starts_ends: (optional) list of tuples (start, end) in seconds for each ground file.
    It should has same length as ground_filenames
    Additions arguments are given to loader_function
    """
    if starts_ends is None:
        starts_ends = [(None, None)] * len(ground_filenames)

    # Ground sessions labels
    yground_by_session = [loader_function(filename, start=s, end=e, **kwarg)
                          for filename, (s, e) in zip(ground_filenames, starts_ends)]
    # Prediction sessions labels
    ypred_by_session = [loader_function(filename, start=s, end=e, **kwarg)
                        for filename, (s, e) in zip(prediction_filenames, starts_ends)]

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


def complete_report(csv_report_filename, labels_of_interest, labels_of_predictors, loader_function,
                    ground_filenames, *argv_prediction_filenames, display=True, starts_ends=None, **kwargs):
    """
    :param csv_report_filename: file to store results. If it is a an empty string it will save no file.
    :param labels_of_predictors: names of predictors to assign to predictions
    :param loader_function: function to load
    :param labels_of_interest: list of labels of interest (among the ones within given files)
    :param ground_filenames: list of filenames for the ground truth
    :param argv_prediction_filenames: lists of filenames for each method of prediction
    (each list is given as a separated argument)
    :param starts_ends: (optional) list of tuples (start, end) in seconds for each ground file.
    It should has same length as ground_filenames
    :param display: True (default) plot results
    :type kwargs: extra arguments for loader function
    """
    # Get scores for each pair of ground and prediction files, for each activities of interest
    scored_sessions = [get_sessions_scores(ground_filenames, prediction_filenames,
                                           loader_function, labels_of_interest, starts_ends=starts_ends, **kwargs)
                       for prediction_filenames in argv_prediction_filenames]

    # Add name of predictors to scored sessions
    for lab, ss in zip(labels_of_predictors, scored_sessions):
        ss.insert(len(ss.columns), "predictor_name", lab)

    complete_report_df = pd.concat(scored_sessions, ignore_index=True)
    if csv_report_filename:
        complete_report_df.to_csv(csv_report_filename, index=None)

    if display:
        display_report(complete_report_df)

    return complete_report_df


def display_report(complete_report_df):
    report_activity_grouped = complete_report_df.groupby("activity")

    for activity_label, single_activity_report in report_activity_grouped:
        print("\n================", activity_label, "================\n")

        plotter.plot_spider_from_report(single_activity_report)
        plotter.plot_violinplot_from_report(single_activity_report)
        plotter.print_f1scores_from_report(single_activity_report)
        plotter.print_time_errors_from_report(single_activity_report)


def load_txt_labels(filename, start=None, end=None, verbose=True, activities_of_interest=None):
    """
    Load activity labels given in filename and return a pandas Series compatible with the scorer.


    :param filename: path to the filename
    :param start: in seconds
    :param end:
    :param verbose:
    :param activities_of_interest:
    :return:
    """
    df = pd.read_table(filename, decimal=',', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ["start", "end", "label"]

    df[["start", "end"]] = df[["start", "end"]].astype('float')
    df = df.round(0)
    df[["start", "end"]] = df[["start", "end"]].astype('int')

    df.label = df.label.str.strip().str.upper()

    # It will modify the limits of partially selected labels
    # Given end and start may be in the middle of a label
    if start:
        df = df[df.end > start]
        df.loc[df.start < start, "start"] = start
        df = df[df.start >= start]
    if end:
        df = df[df.start < end]
        df.loc[df.end > end, "end"] = end
        df = df[df.end <= end]
    # names_of_interest = _names_of_interest
    if verbose:
        print("Labels in (", start, ",", end, ") from", filename, "\n", df.label.unique())

    if activities_of_interest:
        activities_of_interest = [act.strip().upper() for act in activities_of_interest]
        df = df.loc[df.label.isin(activities_of_interest)]

    segments = [Segment(start, end, label) for name, (start, end, label) in df.iterrows()]
    indexes = [np.arange(start, end) for name, (start, end, label) in df.iterrows()]
    if len(segments) < 1:
        print("Warning, you are trying to load a span with no labels from:", filename)

    frames = segments2frames(segments)
    indexes = np.concatenate(indexes)
    serie = pd.Series(frames, index=indexes)

    if serie.index.has_duplicates:
        print("Overlapping labels were found in", filename)
        print("Check labels corresponding to times given below (in seconds):")
        print(serie.index[serie.index.duplicated()])

    s_formatted = serie.reindex(np.arange(serie.index[-1]), fill_value="")

    return s_formatted
