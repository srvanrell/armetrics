
import pandas as pd
from har_utils import *
from models import Segment

standardized_names = {"RUMIA PASTURA": "RUMIA"}
names_of_interest = ["PASTOREO", "RUMIA"]


def load_chewbite(filename):
    df = pd.read_table(filename, decimal=',', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = ["start", "end", "label"]

    df = df.round(0)
    df.label = df.label.str.strip().str.upper()
    print(df.label.unique())
    df.label.replace(standardized_names, inplace=True)
    df = df.loc[df.label.isin(names_of_interest)]

    df[["start", "end"]] = df[["start", "end"]].astype('int64')

    segments = [Segment(start, end, label) for name, (start, end, label) in df.iterrows()]
    indexes = [np.arange(start, end) for name, (start, end, label) in df.iterrows()]

    frames = segments2frames(segments)
    indexes = np.concatenate(indexes)

    s = pd.Series(frames, index=indexes)

    s_formatted = s.reindex(np.arange(s.index[-1]), fill_value="")

    return s_formatted

ground_filename = "data/ground.txt"
prediction_filename = "data/prediction.txt"

ground = load_chewbite(ground_filename)
prediction = load_chewbite(prediction_filename)

print(ground)
print(prediction)

scored_sessions = get_sessions_scores([ground], [prediction], names_of_interest)

print(scored_sessions.groupby("activity").mean())

spider_df_summaries([scored_sessions.groupby("activity")],
                    ["test"])

