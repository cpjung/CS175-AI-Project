import csv
import pandas as pd

quests = pd.read_csv("quests.csv")
print(list(quests.columns.values))
details = quests.Details
rawStr = ""
for d in details:
    rawStr += str(d)
