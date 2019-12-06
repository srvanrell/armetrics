# This file should be run in the 'examples' folder

from armetrics import utils
from armetrics import plotter

ground_filenames = ["./data/ground.txt",
                    "./data/ground.txt"]
prediction_filenames1 = ["./data/test11.txt",
                         "./data/test12.txt"]
prediction_filenames2 = ["./data/test21.txt",
                         "./data/test22.txt"]

standardized_names = {"RUMIA PASTURA": "RUMIA", "PASTURA": "PASTOREO"}
regularity_replacements = {"RUMIA": "REGULAR", "PASTOREO": "REGULAR"}
_names_of_interest = ["PASTOREO", "RUMIA"]


def load_chewbite(filename, start=None, end=None, verbose=True, to_regularity=False):
    from armetrics.models import Segment
    from armetrics.utils import segments2frames
    import pandas as pd
    import numpy as np

    df = pd.read_table(filename, decimal=',', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ["start", "end", "label"]

    df[["start", "end"]] = df[["start", "end"]].astype('float')

    df = df.round(0)
    df.label = df.label.str.strip().str.upper()
    df.label.replace(standardized_names, inplace=True)
    df[["start", "end"]] = df[["start", "end"]].astype('int')

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
    names_of_interest = _names_of_interest
    if to_regularity:
        names_of_interest = ["REGULAR"]
        df.label.replace(regularity_replacements, inplace=True)
    if verbose:
        print("Labels in (", start, ",", end, ") from", filename, "\n", df.label.unique())

    df = df.loc[df.label.isin(names_of_interest)]

    segments = [Segment(start, end, label) for name, (start, end, label) in df.iterrows()]
    indexes = [np.arange(start, end) for name, (start, end, label) in df.iterrows()]
    if len(segments) < 1:
        print("Warning, you are trying to load a span with no labels from:", filename)

    frames = segments2frames(segments)
    indexes = np.concatenate(indexes)

    s = pd.Series(frames, index=indexes)

    if s.index.has_duplicates:
        print("Overlapping labels were found in", filename)
        print("Check labels corresponding to times given below (in seconds):")
        print(s.index[s.index.duplicated()])

    s_formatted = s.reindex(np.arange(s.index[-1]), fill_value="")

    print (s_formatted)

    return s_formatted


def complete_report(loader_function, labels_of_interest, labels_of_predictors,
                    ground_filenames, *argv_prediction_filenames):
    """
    :param loader_function: function to load
    :param labels_of_interest: list of labels of interest (among the ones within given files)
    :param ground_filenames: list of filenames for the ground truth
    :param argv_prediction_filenames: lists of filenames for each method of prediction
    (each list is given as a separated argument)
    """
    ground = [loader_function(filename) for filename in ground_filenames]
    predictions = []
    for prediction_filenames in argv_prediction_filenames:
        predictions.append([loader_function(filename) for filename in prediction_filenames])

    scored_sessions = [utils.get_sessions_scores(ground, pred, labels_of_interest) for pred in predictions]

    plotter.spider_and_violinplot_df_summaries([scored_s.groupby("activity") for scored_s in scored_sessions],
                                               labels_of_predictors)


label_of_interest = ["RUMIA"]
names_of_predictors = ["test1", "test2"]
complete_report(load_chewbite, label_of_interest, names_of_predictors,
                ground_filenames, prediction_filenames1, prediction_filenames2)
