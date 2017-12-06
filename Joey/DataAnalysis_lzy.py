import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import datetime
from scipy import stats

# transform timestamp to datetime
def DealWithTime(data):
    data['time'] = pd.to_datetime(data['timestamp'])
    return data


def GetData(filename):
    data = pd.read_csv(filename)
    data = data[['loggingTime(txt)', 'motionUserAccelerationX(G)', 'motionUserAccelerationY(G)',
                 'motionUserAccelerationZ(G)', 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)',
                 'gyroRotationZ(rad/s)', 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']]
    colums_name = ['timestamp', 'accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'yaw', 'roll', 'pitch']
    data.columns = colums_name
    return data

# a glance at the data
def Overview(data):
    plt.figure(figsize=(15, 20))
    p1 = plt.subplot(3, 1, 1)
    p2 = plt.subplot(3, 1, 2)
    p3 = plt.subplot(3, 1, 3)
    p1.plot(data['time'], data['accX'])
    p2.plot(data['time'], data['accY'])
    p3.plot(data['time'], data['accZ'])
    plt.show()

    plt.figure(figsize=(15, 20))
    p4 = plt.subplot(3, 1, 1)
    p5 = plt.subplot(3, 1, 2)
    p6 = plt.subplot(3, 1, 3)
    p4.plot(data['time'], data['gyroX'])
    p5.plot(data['time'], data['gyroY'])
    p6.plot(data['time'], data['gyroZ'])
    plt.show()

# butter filter
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def butter_lowpass(highcut, fs, order=3):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = signal.butter(order, high, btype='lowpass')
    return b, a


def butter_lowpass_filter(data, highcut, fs, order=3):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


def getStepFrequency(data, fs):
    ffted = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(ffted))
    idx = np.argmax(np.abs(ffted))
    freq = freqs[idx]
    return abs(freq * fs)

# sampling frequency
fs = 30

# find the "mid_stance" point of each step
def seperateEachStep(data, step_cycle):
    steps = pd.DataFrame()
    idx_list = []
    for i in range(0, data.shape[0], step_cycle):
        idx = np.argmax(data['gyroZ'].iloc[i:i + step_cycle])
        if (idx != data.shape[0] - 1):
            idx_list.append(idx)
    for i in range(0, len(idx_list) - 1):
        prev = idx_list[i]
        nxt = idx_list[i + 1]
        while (data['gyroZ'].iloc[prev] > data['gyroZ'].iloc[prev + 1]):
            prev += 1
        while (data['gyroZ'].iloc[nxt] > data['gyroZ'].iloc[nxt - 1]):
            nxt -= 1
        floor1 = prev
        floor2 = nxt
        if (floor1 < floor2):
            idx = np.argmax(data['gyroZ'].iloc[floor1:floor2])
            steps = steps.append(data.iloc[idx])
    return steps

# calculate Vx and Vy with correction
def CorrectedV(begin, end, deltaVy, deltaVx, deltaT, initVy, initVx, data):
    data.set_value(begin, 'Vx', initVx)
    data.set_value(begin, 'Vy', initVy)
    for i in range(int(begin + 1), int(end)):
        data.set_value(i, 'Vx', data.loc[i-1,'Vx']+ (data.loc[i-1,'accX'] + deltaVx/deltaT)/fs)
        data.set_value(i, 'Vy', data.loc[i-1,'Vy']+ (data.loc[i-1,'accY'] + deltaVy/deltaT)/fs)

# calculate displacement of X and Y simply by integration
def calcY(data, begin, end):
    return data.loc[begin:end, 'Vy'].sum() / fs

def calcX(data, begin, end):
    return data.loc[begin:end, 'Vx'].sum() / fs

# Y correction
def CorrectedY(begin, end, calcY, deltaT, data):
    data.set_value(begin, 'correctY', 0)
    for i in range(int(begin + 1), int(end)):
        data.set_value(i, 'correctY', data.loc[i - 1, 'correctY'] + (data.loc[i - 1, 'Vy'] - calcY/deltaT) / fs)

# get foot clearance(height) my get the maxinum y displacement(which is in the negative direction)
def getFC(data, begin, end):
    idx = np.argmin(data.loc[begin:end, 'correctY'])
    return data.loc[idx, 'correctY']

# calculation V simply by integration
def calcSpeedY(data, begin, end):
    return data.loc[begin:end, 'accY'].sum() / fs


def calcSpeedZ(data, begin, end):
    return data.loc[begin:end, 'accZ'].sum() / fs


def calcSpeedX(data, begin, end):
    return data.loc[begin:end, 'accX'].sum() / fs

label = 'right '

def process_data(left):
    plt.figure(figsize=(16, 9))
    p1 = plt.subplot(2, 2, 1)
    p1.plot(left['accX'])
    plt.title(label + ' accX before FFT')

    p2 = plt.subplot(2, 2, 2)
    p2.plot(left['gyroZ'])
    plt.title(label + ' gyroZ before FFT')

    highcut = 5.5
    left['accX'] = butter_lowpass_filter(left['accX'], highcut, fs)
    left['accY'] = butter_lowpass_filter(left['accY'], highcut, fs)
    left['accZ'] = butter_lowpass_filter(left['accZ'], highcut, fs)

    lowcut = 0.08
    highcut = 6
    left['gyroX'] = butter_bandpass_filter(left['gyroX'], lowcut, highcut, fs)
    left['gyroY'] = butter_bandpass_filter(left['gyroY'], lowcut, highcut, fs)
    left['gyroZ'] = butter_bandpass_filter(left['gyroZ'], lowcut, highcut, fs)

    step_frequency = getStepFrequency(left['accX'], fs)

    print(label + ' frequency: ', end=' ')
    print(step_frequency)

    step_cycle = int(round(fs / step_frequency))

    steps = seperateEachStep(left, step_cycle)

    p3 = plt.subplot(2, 2, 3)
    p3.plot(steps['accX'], 'r*', left['accX'], 'b-')
    plt.title(label + ' accX after FFT with step separator')

    p4 = plt.subplot(2, 2, 4)
    p4.plot(steps['gyroZ'],'r*',left['gyroZ'], 'b-')
    plt.title(label + ' gyroZ after FFT with step separator')
    plt.savefig(label + ' comparation_before_after_Fast_Fourier_Transform')

    plt.clf()

    # assume at mid stance the cell phone is vertical and the it only have Vx due to rotation around ankle
    steps['initVy'] = 0
    steps['initVx'] = 0.1 * steps['gyroZ']
    steps['endVy'] = steps['initVy'].shift(-1)
    steps['endVx'] = steps['initVx'].shift(-1)

    steps['begin_idx'] = steps.index
    steps['end_idx'] = steps['begin_idx'].shift(-1)
    steps = steps.dropna()
    steps['end_idx'] = steps['end_idx'].astype(int)

    # get Vs by integration
    steps['calcVy'] = steps.apply((lambda row: calcSpeedY(left, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcVz'] = steps.apply((lambda row: calcSpeedZ(left, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcVx'] = steps.apply((lambda row: calcSpeedX(left, row['begin_idx'], row['end_idx'])), axis=1)

    # the shift of Vs
    steps['deltaVy'] = steps['endVy'] - steps['calcVy']
    steps['deltaVx'] = steps['endVx'] - steps['calcVx']
    steps['deltaT'] = (steps['end_idx'] - steps['begin_idx']) / fs
    steps = steps.reset_index(drop=True)

    begin = steps['begin_idx'].iloc[0]
    end = steps['end_idx'].iloc[-1]
    left_data = left[begin: end]

    # correct the shift
    steps.apply((lambda row: CorrectedV(row['begin_idx'], row['end_idx'], row['deltaVy'], row['deltaVx'], row['deltaT'],
                                        row['initVy'], row['initVx'], left_data)), axis=1)

    # get displacements
    steps['disY'] = steps.apply((lambda row: calcY(left_data, row['begin_idx'], row['end_idx'])), axis=1)
    steps['disX'] = steps.apply((lambda row: calcX(left_data, row['begin_idx'], row['end_idx'])), axis=1)

    # correct displacement of Y
    steps.apply((lambda row: CorrectedY(row['begin_idx'], row['end_idx'], row['disY'], row['deltaT'], left_data)),
                axis=1)
    steps['FC'] = steps.apply((lambda row: getFC(left_data, row['begin_idx'], row['end_idx'])), axis=1)

    plt.plot(left_data['time'], left_data['Vy'], 'b-')
    plt.plot(steps['time'], steps['initVy'], 'r*')
    plt.xlabel('time')
    plt.ylabel('Vy(m/s)')
    plt.title(label + ' Vy')
    plt.savefig(label + ' Vy')
    plt.clf()

    plt.plot(left_data['time'], left_data['Vx'], 'b-')
    plt.plot(steps['time'], steps['initVx'], 'r*')
    plt.xlabel('time')
    plt.ylabel('Vx(m/s)')
    plt.savefig(label + ' Vx')
    plt.clf()

    return steps

OUTPUT_TEMPLATE = (
    "Test summary\n\n"
    "Testing X displacement of each step...\n"
    "left normal test pvalue: {left_dis_normal_p:.3g}\n"
    "right normal test pvalue: {right_dis_normal_p:.3g}\n"
    "ttest pvalue: {dis_ttest_p:.3g}\n\n"
    "Testing duration of each step...\n"
    "left normal test pvalue: {left_dur_normal_p:.3g}\n"
    "right normal test pvalue: {right_dur_normal_p:.3g}\n"
    "ttest pvalue: {dur_ttest_p:.3g}\n\n"
    "mann whitney u test of X displacement of left and right: {dis_mann_p:.3g}\n"
    "mann whitney u test of duration of left and right: {dur_mann_p:.3g}\n\n"
    "left walking speed: {left_v:.3g}\n"
    "right walking speed: {right_v:.3g}\n"
    "actual walking speed: {actual_v:.3g}\n"
)

def main():
    right = GetData('../dataset/lzy_right_2017-12-05_15-37-31_-0800.csv')
    right = DealWithTime(right)
    # the unit of acceleration is g, here we transform it to m/s2
    right['accY'] *= 9.81
    right['accX'] *= 9.81
    right['accZ'] *= 9.81
    # cut the data by the walking interval
    right = right[right['time'] > datetime.datetime(2017, 12, 5, 23, 37, 52, 200000)]
    right = right[right['time'] < datetime.datetime(2017, 12, 5, 23, 38, 6, 700000)]
    right = right.reset_index(drop=True)

    steps_right = process_data(right)

    global label
    label = 'left'
    left = GetData('../dataset/lzy_left_2017-12-05_15-37-28_-0800.csv')
    left = DealWithTime(left)
    left['accX'] *= 9.81
    left['accY'] *= 9.81
    left['accZ'] *= 9.81
    left = left[left['time'] > datetime.datetime(2017, 12, 5, 23, 37, 52, 700000)]
    left = left[left['time'] < datetime.datetime(2017, 12, 5, 23, 38, 7, 000000)]

    left = left.reset_index(drop=True)
    steps_left = process_data(left)

    steps_left.to_csv('steps_left.csv')
    steps_right.to_csv('steps_right.csv')

    print(OUTPUT_TEMPLATE.format(
        left_dis_normal_p = stats.normaltest(steps_left['disX']).pvalue,
        right_dis_normal_p = stats.normaltest(steps_right['disX']).pvalue,
        dis_ttest_p = stats.ttest_ind(steps_left['disX'], steps_right['disX']).pvalue,
        left_dur_normal_p = stats.normaltest(steps_left['deltaT']).pvalue,
        right_dur_normal_p = stats.normaltest(steps_right['deltaT']).pvalue,
        dur_ttest_p = stats.ttest_ind(steps_left['deltaT'], steps_right['deltaT']).pvalue,
        dis_mann_p = stats.mannwhitneyu(steps_left['disX'], steps_right['disX']).pvalue,
        dur_mann_p = stats.mannwhitneyu(steps_left['deltaT'], steps_right['deltaT']).pvalue,
        left_v = -steps_left['disX'].sum() / steps_left['deltaT'].sum(),
        right_v = -steps_right['disX'].sum() / steps_right['deltaT'].sum(),
        actual_v = 19/16
    ))

if __name__ == '__main__':
    main()
