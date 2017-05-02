from har_utils import *

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
ground = events2frames(ground_ev, last_index=31)

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
output1 = events2frames(output1_ev, last_index=31)

print("ground:", ground)
print("sysout:", output1)

scored_segments = frames2segments(ground, output1)
for seg in scored_segments:
    print(seg)

# labeled = frames2segments(ground, output1)
# a = labeled_segments2labeled_frames(labeled)
# print(a, len(a), len(ground))

for k in score_events(scored_segments, ground_ev, output1_ev):
    print(k)
# print(labeled_segments2labeled_events(labeled_seg, output1_ev))


# Loading a label from hasc database

import pandas as pd
import numpy as np
import utils

# features_folder = 'features/'

window_size = 512
window_overlap = 256

crop_starting_seconds = 2.0
crop_ending_seconds = 5.0
crop_str = "_crop_fftxyz"

aux_path = "/home/sebastian/Datasets/HASC-PAC2016/BasicActivity/0_sequence/person01029/HASC0100051-acc.csv"
aux_acc = utils.read_acc(aux_path)
# for aux_win in utils.get_windows(aux_acc, window_size, window_overlap):
#     aux_len = len(features_from_frame(aux_win))
# num_features = aux_len
# num_features


filtering_str = ""  # an empty string won't do anything
# filtering_str = "_right_front_pants_fit"  # comment this line to avoid filtering files

hasc_df = pd.read_pickle('hasc_df.pkl')
hasc_df = hasc_df[pd.notnull(hasc_df.acc_file)]

# to filter right front pocket pants TerminalPosition
if filtering_str:
    for word in filtering_str.split("_"):
        print(word)
        hasc_df = hasc_df[hasc_df["TerminalPosition"].str.contains(word)]

hasc_df_segmented = hasc_df[hasc_df.activity != "sequence"]
hasc_df_sequence = hasc_df[hasc_df.activity == "sequence"]
hasc_df_sequence = hasc_df_sequence[pd.notnull(hasc_df_sequence.label_file)]  # Avoid records without labels file

temp_list = []
# for entry_index in hasc_df_sequence.index[0]:
#     print(entry_index)

acc_filename = hasc_df["acc_file"][0]
print(acc_filename)
labels = utils.read_labels(acc_filename.replace("-acc.csv", ".label"))
print(labels)


window_times = utils.get_window_times(acc.mag, window_size, window_overlap)
window_labels = [utils.get_label(labels.get_values(), acc.t[t_center]) for ti, te, t_center in window_times]

temp_df = pd.DataFrame(feature_frames, columns=["f%d" % d for d in range(num_features)])
temp_df["labels"] = window_labels
temp_df["labels"] = temp_df["labels"].astype("category")
temp_df["subject"] = hasc_df["subject"][entry_index]
temp_df["session"] = hasc_df["session"][entry_index]
temp_df["acc_file"] = hasc_df["acc_file"][entry_index]
