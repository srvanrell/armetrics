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

