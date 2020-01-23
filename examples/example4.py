# This file should be run in the 'examples' folder

from armetrics import utils

filename = "./data/label_track_example.txt"

utils.load_txt_labels(filename, start=3000, end=None)
