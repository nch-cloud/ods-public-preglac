from __future__ import print_function
import sys
import os
from collections import defaultdict
from random import sample

HPO_FILE = sys.argv[1]
PHENOTYPIC_ABNORMALITY_ID = "FPO:0000001"

def print_hpo():
  hpo = open(HPO_FILE)
  for line in hpo:
    if "[Term]" in line:
      hpoID = ""
      continue
    lineData = line.strip().split(": ")
    if len(lineData) < 2: continue
    key = lineData[0]
    value = ": ".join(lineData[1:])
    if key == "id":
      hpoID = value
    if key == "name":
      print(hpoID + "\t" + value)
  hpo.close()

print_hpo()


