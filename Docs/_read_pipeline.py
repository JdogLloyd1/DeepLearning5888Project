import json, io, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

path = r"c:\Users\jonyl\Documents\GitHub\DeepLearning5888Project\Code\LSTM\LSTM baseline.ipynb"
with io.open(path, encoding="utf-8") as f:
    nb = json.load(f)

# Dump cells 1-10 to see imports, config, pipeline, and LSTM model
for i in range(1, 11):
    print(f"\n====== CELL {i} ({nb['cells'][i]['cell_type']}) ======")
    print("".join(nb["cells"][i]["source"]))
