
from armetrics.har_utils import *

ground_filename = "./data/ground.txt"
prediction_filename1 = "./data/cbar.txt"
prediction_filename2 = "./data/walter.txt"

standardized_names = {"RUMIA PASTURA": "RUMIA", "PASTURA": "PASTOREO"}
regularity_replacements = {"RUMIA": "REGULAR", "PASTOREO": "REGULAR"}
_names_of_interest = ["PASTOREO", "RUMIA"]


def load_chewbite(filename, start=None, end=None, verbose=True, to_regularity=False):
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

    return s_formatted

ground = load_chewbite(ground_filename)
prediction1 = load_chewbite(prediction_filename1)
prediction2 = load_chewbite(prediction_filename2)

# print(ground)
# print(prediction)
noi = ["RUMIA"]
scored_sessions1 = get_sessions_scores([ground], [prediction1], noi)
scored_sessions2 = get_sessions_scores([ground], [prediction2], noi)

# print(scored_sessions.groupby("activity").mean())

spider_df_summaries([scored_sessions1.groupby("activity"),
                     scored_sessions2.groupby("activity")],
                    ["test1", "test2"])

