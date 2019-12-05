from armetrics.utils import *

# Example taken from Figure 2 of the paper
ground_ev = [[1, 6],
             [8, 11],
             [14, 15],
             [16, 17],
             [18, 19],
             [20, 21],
             [22, 24],
             [25, 30]
             ]
ground = events2frames(ground_ev, length=31)

output1_ev = [[1, 2],
              [3, 4],
              [5, 7],
              [9, 10],
              [11, 12],
              [13, 15],
              [18, 21],
              [22, 26],
              [27, 28],
              [29, 30]
              ]
output1 = events2frames(output1_ev, length=31)

print("ground:", ground)
print("sysout:", output1)

scored_segments = frames2segments(ground, output1)
for seg in scored_segments:
    print(seg)

scored_frames = segments2frames(scored_segments)

scored_true_events, scored_pred_events = score_events(scored_segments, ground_ev, output1_ev)
print(scored_true_events)
print(scored_pred_events)

print(events_summary(scored_true_events, scored_pred_events, normalize=True))
print(frames_summary(scored_frames))


# Loading a label from hasc database

import pandas as pd
import hasc_utils

# features_folder = 'features/'

window_size = 512
window_overlap = 256

crop_starting_seconds = 2.0
crop_ending_seconds = 5.0
crop_str = "_crop_fftxyz"

aux_path = "/home/sebastian/Datasets/HASC-PAC2016/BasicActivity/0_sequence/person01029/HASC0100051-acc.csv"

hasc_df = pd.read_pickle('hasc_df.pkl')
hasc_df = hasc_df[pd.notnull(hasc_df.acc_file)]

hasc_df_segmented = hasc_df[hasc_df.activity != "sequence"]
hasc_df_sequence = hasc_df[hasc_df.activity == "sequence"]
hasc_df_sequence = hasc_df_sequence[pd.notnull(hasc_df_sequence.label_file)]  # Avoid records without labels file

acc_filename = hasc_df["acc_file"][0]
acc = hasc_utils.read_acc(acc_filename)
labels = hasc_utils.read_labels(acc_filename.replace("-acc.csv", ".label"))

window_times = hasc_utils.get_window_times(acc.mag, window_size, window_overlap)
window_labels = [hasc_utils.get_label(labels.get_values(), acc.t[t_center]) for ti, te, t_center in window_times]

print(acc_filename)
print(labels)
print(window_labels)
print(acc.t.iloc[0], acc.t.iloc[-1])

stay_true_bin = binarize_frames(window_labels, "walk")
stay_true_evs = binframes2events(stay_true_bin)
stay_pred_evs = [[3, 12],
                 [15, 21],
                 [30, 55],
                 [70, 111]]
stay_pred_bin = events2frames(stay_pred_evs, length=len(stay_true_bin))

print("true ev", stay_true_evs)
print("pred ev", stay_pred_evs)

get_scores(stay_true_bin, stay_pred_bin)
