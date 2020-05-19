# This file should be run in the 'examples' folder

from armetrics import utils

ground_filenames1 = ["./data/ground1.txt",
                     "./data/ground2.txt"]
prediction_filenames1 = ["./data/test11.txt",
                         "./data/test12.txt"]
prediction_filenames2 = ["./data/test21.txt",
                         "./data/test22.txt"]

report_csv = "example5_report.csv"
activities_of_interest = ["ACTIVIDAD_A", "ACTIVIDAD_B"]
names_of_predictors = ["Predictor2 [pos0]", "Predictor1 [pos1]"]  # Predictors are not given in alphabetical order

utils.complete_report(report_csv, activities_of_interest, names_of_predictors, utils.load_txt_labels,
                      ground_filenames1, prediction_filenames1, prediction_filenames2)
