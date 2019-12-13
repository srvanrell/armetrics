from armetrics import utils
from armetrics import scorer

# Example taken from Figure 2 of the paper
ground_ev = [[2, 4],
             ]
ground = utils.events2frames(ground_ev, length=10)

output1_ev = []
output1 = utils.events2frames(output1_ev, length=10)

print("ground:", ground)
print("sysout:", output1)

scored_segments = utils.frames2segments(ground, output1)
for seg in scored_segments:
    print(seg)

scored_frames = utils.segments2frames(scored_segments)

scored_true_events, scored_pred_events = scorer.score_events(scored_segments, ground_ev, output1_ev)
print(scored_true_events)
print(scored_pred_events)

print("Event summary\n:", scorer.events_summary(scored_true_events, scored_pred_events, normalize=True))
print(scorer.frames_summary(scored_frames))
