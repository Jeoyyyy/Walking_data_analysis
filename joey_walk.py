import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import re
from scipy import stats
import math

def to_time(s):
    pattern = re.compile(r'\d+-\d+-\d+ (\d+):(\d+):(\d+).(\d+) .*')
    m = pattern.match(s)
    if m:
        return float(m.group(1))*10000000 + float(m.group(2))*100000 + float(m.group(3))*1000 + float(m.group(4))
    else:
        return None

def pitch2Rx(pitch):
    Rx = [[1, 0, 0],
          [0, math.cos(pitch), -math.sin(pitch)],
          [0, math.sin(pitch), math.cos(pitch)]]
    return Rx

def roll2Ry(roll):
    Ry = [[math.cos(roll), 0, math.sin(roll)],
          [0, 1, 0],
          [-math.sin(roll), 0, math.cos(roll)]]
    return Ry

def yaw2Rz(yaw):
    Rz = [[math.cos(yaw), -math.sin(yaw), 0],
          [math.sin(yaw), math.cos(yaw), 0],
          [0, 0, 1]]
    return Rz

b, a = signal.butter(3, 0.25, btype='lowpass', analog=False)

data_left = pd.read_csv('./dataset/lzy_left_2017-11-27_14-36-05_-0800.csv')
data_right = pd.read_csv('./dataset/lzy_right_2017-11-27_14-36-02_-0800.csv')

a_left_cut = pd.DataFrame({'timestamp': data_left['motionTimestamp_sinceReboot(s)'],
                           'time': data_left['loggingTime(txt)'],
                            'ax': data_left['motionUserAccelerationX(G)'],
                            'ay': data_left['motionUserAccelerationY(G)'],
                            'az': data_left['motionUserAccelerationZ(G)'],
                           'pitch': data_left['motionPitch(rad)'],
                           'roll': data_left['motionRoll(rad)'],
                           'yaw': data_left['motionYaw(rad)']})
a_left_cut['time'] = a_left_cut['time'].apply(to_time)
a_left_cut['timestamp_shift'] = a_left_cut['timestamp'].shift(-1)
a_left_cut['delta_time'] = a_left_cut['timestamp_shift'] - a_left_cut['timestamp']
a_left_cut = a_left_cut[(a_left_cut['time'] > 143632500) &
                        (a_left_cut['time'] < 143648000)].reset_index()
a_left_cut['index'] -= 785
a_left_cut['ax'] = signal.filtfilt(b, a, a_left_cut['ax'])
a_left_cut['ay'] = signal.filtfilt(b, a, a_left_cut['ay'])
a_left_cut['az'] = signal.filtfilt(b, a, a_left_cut['az'])

# a_left_cut['Rx'] = a_left_cut['pitch'].apply(pitch2Rx)
# a_left_cut['Ry'] = a_left_cut['roll'].apply(roll2Ry)
# a_left_cut['Rz'] = a_left_cut['yaw'].apply(yaw2Rz)
# a_left_cut['Gax'] = -(np.cos(a_left_cut['yaw'])*a_left_cut['ax'] + \
#     -np.cos(a_left_cut['pitch'])*np.sin(a_left_cut['yaw'])*a_left_cut['ay'] + \
#     np.sin(a_left_cut['pitch'])*np.sin(a_left_cut['yaw'])*a_left_cut['az'])
a_left_cut['Gax'] = -(np.cos(a_left_cut['yaw'])*a_left_cut['ax'] - np.sin(a_left_cut['yaw'])*a_left_cut['ay'])

plt.plot(a_left_cut['time'], a_left_cut['Gax'], '-')

a_left_cut['vx'] = a_left_cut['delta_time'] * a_left_cut['Gax']
for idx, row in a_left_cut.iterrows():
    if idx == 0:
        continue
    if idx%33 == 0:
        a_left_cut.set_value(idx, 'vx', a_left_cut.iloc[idx - 1]['vx'] + a_left_cut.iloc[idx]['vx'])
    else:
        a_left_cut.set_value(idx, 'vx', a_left_cut.iloc[idx-1]['vx']+a_left_cut.iloc[idx]['vx'])

print(a_left_cut)

plt.plot(a_left_cut['time'], a_left_cut['vx'], '-')
plt.show()

ax_left_filt = signal.filtfilt(b, a, a_left_cut['ax'])
# plt.plot(a_left_cut['time'], a_left_cut['ax'], 'b-')
# plt.plot(a_left_cut['time'], ax_left_filt, 'r-')
# plt.show()

ay_left_filt = signal.filtfilt(b, a, a_left_cut['ay'])

# plt.plot(a_left_cut['time'], a_left_cut['ay'], 'r-')
# plt.plot(a_left_cut['time'], ay_left_filt, 'r-')
# plt.show()

az_left_filt = signal.filtfilt(b, a, a_left_cut['az'])
# plt.plot(a_left_cut['time'], a_left_cut['az'], 'b-')
# plt.plot(a_left_cut['time'], az_left_filt, 'r-')
# plt.show()

a_right_cut = pd.DataFrame({'timestamp': data_right['motionTimestamp_sinceReboot(s)'],
                            'time': data_right['loggingTime(txt)'],
                            'ax': data_right['motionUserAccelerationX(G)'],
                            'ay': data_right['motionUserAccelerationY(G)'],
                            'az': data_right['motionUserAccelerationZ(G)']})
a_right_cut['time'] = a_right_cut['time'].apply(to_time)
a_right_cut = a_right_cut[(a_right_cut['time'] > 143631000) &
                          (a_right_cut['time'] < 143651000)].reset_index()
a_right_cut['index'] -= 932
a_right_cut = a_right_cut[a_right_cut['index'] <= 625]

ax_right_filt = signal.filtfilt(b, a, a_right_cut['ax'])
# plt.plot(a_right_cut['time'], a_right_cut['ax'], 'r-')
# plt.plot(a_right_cut['time'], ax_right_filt, 'g-')
# plt.show()

ay_right_filt = signal.filtfilt(b, a, a_right_cut['ay'])

# plt.plot(a_right_cut['time'], a_right_cut['ay'], 'b-')
# plt.plot(a_right_cut['time'], ay_right_filt, 'g-')
# plt.show()

az_right_filt = signal.filtfilt(b, a, a_right_cut['az'])
# plt.plot(a_right_cut['time'], a_right_cut['az'], 'b-')
# plt.plot(a_right_cut['time'], az_right_filt, 'g-')
# plt.show()

# print(stats.normaltest(ay_left_filt).pvalue)
# print(stats.normaltest(ay_right_filt).pvalue)
# print(stats.ttest_ind(ay_left_filt, ay_right_filt).pvalue)
# print(stats.mannwhitneyu(ay_left_filt, ay_right_filt).pvalue)

# contingency = np.stack([np.abs(ay_left_filt), np.abs(ay_right_filt)])
# print(contingency)
# print(stats.chi2_contingency(contingency)[1])
pace = 120    # step/minute