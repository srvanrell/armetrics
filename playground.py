from har_utils import *

# Example taken from Figure 2 of the paper
ground_ev = [[1, 6],
             [8, 11],
             [14, 15],
             [16, 17],
             [18, 19],
             [20, 21],
             [22, 24],
             [25, 30]]
ground = events2frames(ground_ev, end=31)

output1_ev = [[1, 2],
              [3, 4],
              [5, 7],
              [9, 10],
              [11, 12],
              [13, 15],
              [18, 21],
              [22, 26],
              [27, 28],
              [29, 30]]
output1 = events2frames(output1_ev, end=31)

print("ground:", ground)
print("sysout:", output1)

labeled_seg = frames2segments(ground, output1)
for start, end, label in labeled_seg:
    print(start, end, label)

# labeled = frames2segments(ground, output1)
# a = labeled_segments2labeled_frames(labeled)
# print(a, len(a), len(ground))

for k in labeled_segments2labeled_events(labeled_seg, ground_ev, output1_ev):
    print(k)
# print(labeled_segments2labeled_events(labeled_seg, output1_ev))

