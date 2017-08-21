
from armetrics.har_utils import *

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

