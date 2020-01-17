# This file should be run in the 'examples' folder

from armetrics import utils
from armetrics import plotter

ground_filenames = ["./data/ground1.txt",
                    "./data/ground2.txt"]
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

    # print (s_formatted)
    # FIXME label column could be categorical

    return s_formatted


labels_of_interest = ["RUMIA", "PASTOREO"]
scored_sessions1 = utils.get_sessions_scores(ground_filenames, prediction_filenames1, load_chewbite, labels_of_interest)
scored_sessions2 = utils.get_sessions_scores(ground_filenames, prediction_filenames2, load_chewbite, labels_of_interest)

plotter.spider_and_violinplot_df_summaries([scored_sessions1.groupby("activity"),
                                            scored_sessions2.groupby("activity")],
                                           ["test1", "test2"])
