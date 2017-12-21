# CMPT 318 Project
# Project Topic: Sensors, Noise, and Walking
## Group Name: Stark Industries

## Group Member:
* ZHIZHOU JIANG zhizhouj@sfu.ca
* Manqing Zhu manqingz@sfu.ca
* Zhaoyang Li zla143@sfu.ca

## File organization
* There are 3 .py source file in 3 directories named Joey, Tony and Margaret.
* Each .py file is used to analize a set of data composed of data recorded from left and right foot
* The algorithm used in the 3 file to do the analsis is the same, but there are 3 files because each dataset need to be extracted and cleaned a little differently
* Datasets used by the programs is in dataset directory

## How to run:
* Goto each of the 3 directories and run the .py files with no argument. Order does not matter.
```
python3 DataAnalysis_zmq.py
python3 DataAnalysis_tony.py
python3 DataAnalysis_lzy.py
```

## Required Libraries
* numpy
* pandas
* matplotlib.pyolot
* scipy
* datetime

## Expected output:
* The source files should print out the result of walking pace, walking speed, and stats test summary
* They also should create some figures in the current directory and .csv file containing the calculated data result into the dataset directory


