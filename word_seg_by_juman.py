#!/usr/bin/env python3

import sys
import gzip
import csv
from pyknp import Juman
juman = Juman(command="/raid/users/nakazawa/usr/bin/juman")

with gzip.open(sys.argv[1], "rt") as f_in, open(sys.argv[1]+".new.txt", "w") as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    for row in reader:
        result = juman.analysis(row[1])
        row[1] = " ".join([mrph.midasi for mrph in result])
        writer.writerow(row)
