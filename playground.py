from har_utils import *

# ground = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0])
ground = events2frames([[2, 4],
                        [7, 8]],
                       end=10)
# output1 = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
output1 = events2frames([[0, 0],
                         [3, 3],
                         [15, 17]],
                        end=10)

print("ground:", ground)
print("sysout:", output1)

for start, end, label in frames2segments(ground, output1):
    print(start, end, label)

labeled = frames2segments(ground, output1)
a = labeled_segments2labeled_frames(labeled)
print(a, len(a), len(ground), labeled)