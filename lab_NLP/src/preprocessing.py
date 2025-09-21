import pandas as pd
import os

with open("./data/raw/medicheck-expert.csv") as f:
    for i, line in enumerate(f):
        print(i, line.strip())
        if i > 15:
            break

