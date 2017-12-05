import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import datetime
import re
from scipy import stats


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

    # left = GetData("../dataset/lzy_left_2017-11-27_14-36-05_-0800.csv")
    # right =  GetData("../dataset/lzy_right_2017-11-27_14-36-02_-0800.csv")
fs = 30

def seperateEachStep(data, step_cycle):
    steps = pd.DataFrame()
    idx_list = []
    for i in range(0, data.shape[0], step_cycle):
        idx = np.argmax(data['gyroZ'].iloc[i:i + step_cycle])
        if (idx != data.shape[0] - 1):
            idx_list.append(idx)
            #         steps = steps.append(data.iloc[idx])
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


def CorrectedV(begin, end, deltaVy, deltaVx, deltaT, initVy, initVx, data):
    data.set_value(begin, 'Vx', initVx)
    data.set_value(begin, 'Vy', initVy)
    for i in range(int(begin + 1), int(end)):
        data.set_value(i, 'Vx', data.loc[i-1,'Vx']+ (data.loc[i-1,'accX'] + deltaVx/deltaT)/fs)
        data.set_value(i, 'Vy', data.loc[i-1,'Vy']+ (data.loc[i-1,'accY'] + deltaVy/deltaT)/fs)


def calcY(data, begin, end):
    return data.loc[begin:end, 'Vy'].sum() / fs

def calcX(data, begin, end):
    return data.loc[begin:end, 'Vx'].sum() / fs
    # data.set_value(begin, 'disX', 0)
    # for i in range(int(begin+1), int(end)):
    #     data.set_value(i, 'disX', data.loc[i - 1, 'disX'] + data.loc[i-1, 'Vx'] / fs)
    # return data.loc[end-1, 'disX']

def CorrectedY(begin, end, calcY, deltaT, data):
    data.set_value(begin, 'correctY', 0)
    for i in range(int(begin + 1), int(end)):
        data.set_value(i, 'correctY', data.loc[i - 1, 'correctY'] + (data.loc[i - 1, 'Vy']) / fs)


def getFC(data, begin, end):
    idx = np.argmin(data.loc[begin:end, 'correctY'])
    return data.loc[idx, 'correctY']


def calcSpeedY(data, begin, end):
    return data.loc[begin:end, 'accY'].sum() / fs


def calcSpeedZ(data, begin, end):
    return data.loc[begin:end, 'accZ'].sum() / fs


def calcSpeedX(data, begin, end):
    return data.loc[begin:end, 'accX'].sum() / fs

# def calcSpeedY(data, begin, end, initVy):
#     data.set_value(begin, 'Vy', initVy)
#     for i in range(int(begin + 1), int(end)):
#         data.set_value(i, 'Vy', data.loc[i - 1, 'Vy'] +
#                        (data.loc[i - 1, 'accY'] * np.cos(data.loc[i - 1, 'yaw']) +
#                         data.loc[i - 1, 'accX'] * np.sin(data.loc[i - 1, 'yaw'])) / fs)
#
#     return data.loc[end - 1, 'Vy']
#
# def calcSpeedX(data, begin, end, initVx):
#     data.set_value(begin, 'Vx', initVx)
#     for i in range(int(begin + 1), int(end)):
#         data.set_value(i, 'Vx', data.loc[i - 1, 'Vx'] +
#                        (data.loc[i - 1, 'accY'] * -np.sin(data.loc[i - 1, 'yaw']) +
#                         data.loc[i - 1, 'accX'] * np.cos(data.loc[i - 1, 'yaw'])) / fs)
#
#     return data.loc[end - 1, 'Vx']

label = 'Zhaoyang\'s right'

def process_data(left):
    plt.plot(left['accX'])
    plt.title(label + ' accX before FFT')
    plt.savefig(label + ' accX before FFT')
    plt.clf()

    plt.plot(left['gyroZ'])
    plt.title(label + ' gyroZ before FFT')
    plt.savefig(label + ' gyroZ before FFT')
    plt.clf()

    highcut = 5
    left['accX'] = butter_lowpass_filter(left['accX'], highcut, fs)
    left['accY'] = butter_lowpass_filter(left['accY'], highcut, fs)
    left['accZ'] = butter_lowpass_filter(left['accZ'], highcut, fs)

    lowcut = 0.08
    highcut = 6
    left['gyroX'] = butter_bandpass_filter(left['gyroX'], lowcut, highcut, fs)
    left['gyroY'] = butter_bandpass_filter(left['gyroY'], lowcut, highcut, fs)
    left['gyroZ'] = butter_bandpass_filter(left['gyroZ'], lowcut, highcut, fs)

    step_frequency = getStepFrequency(left['accX'], fs)

    print('frequency: ', end=' ')
    print(step_frequency)

    step_cycle = int(round(fs / step_frequency))

    steps = seperateEachStep(left, step_cycle)
    plt.plot(steps['gyroZ'],'r*',left['gyroZ'], 'b-')
    plt.title(label + ' gyroZ after FFT')
    plt.savefig(label + ' gyroZ after FFT')
    plt.clf()

    plt.plot(steps['accX'],'r*',left['accX'], 'b-')
    plt.title(label + ' accX after FFT with step separator')
    plt.savefig(label + ' accX after FFT')
    plt.clf()

    steps['initVy'] = 0
    steps['initVx'] = 0.15 * steps['gyroZ']
    # steps['initVy'] = 0.1 * steps['gyroZ'] * np.sin(steps['yaw'])
    # steps['initVx'] = 0.1 * steps['gyroZ'] * np.cos(steps['yaw'])
    steps['endVy'] = steps['initVy'].shift(-1)
    steps['endVx'] = steps['initVx'].shift(-1)

    steps['begin_idx'] = steps.index
    steps['end_idx'] = steps['begin_idx'].shift(-1)
    steps = steps.dropna()
    steps['end_idx'] = steps['end_idx'].astype(int)

    steps['calcVy'] = steps.apply((lambda row: calcSpeedY(left, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcVz'] = steps.apply((lambda row: calcSpeedZ(left, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcVx'] = steps.apply((lambda row: calcSpeedX(left, row['begin_idx'], row['end_idx'])), axis=1)

    steps['deltaVy'] = steps['endVy'] - steps['calcVy']
    steps['deltaVx'] = steps['endVx'] - steps['calcVx']
    steps['deltaT'] = (steps['end_idx'] - steps['begin_idx']) / fs
    steps = steps.reset_index(drop=True)

    begin = steps['begin_idx'].iloc[0]
    end = steps['end_idx'].iloc[-1]
    left_data = left[begin: end]

    steps.apply((lambda row: CorrectedV(row['begin_idx'], row['end_idx'], row['deltaVy'], row['deltaVx'], row['deltaT'],
                                        row['initVy'], row['initVx'], left_data)), axis=1)

    steps['calcY'] = steps.apply((lambda row: calcY(left_data, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcX'] = steps.apply((lambda row: calcX(left_data, row['begin_idx'], row['end_idx'])), axis=1)

    steps.apply((lambda row: CorrectedY(row['begin_idx'], row['end_idx'], row['calcY'], row['deltaT'], left_data)),
                axis=1)
    # plt.plot(left_data['correctY'])
    # plt.title("disY")
    # plt.show()
    steps['FC'] = steps.apply((lambda row: getFC(left_data, row['begin_idx'], row['end_idx'])), axis=1)

    plt.plot(left_data['time'], left_data['Vy'], 'b-')
    plt.plot(steps['time'], steps['initVy'], 'r*')
    plt.xlabel('time')
    plt.ylabel('Vy(m/s)')
    plt.title(label + ' Vy')
    plt.savefig(label + ' Vy')
    plt.clf()
    # plt.plot(steps['FC'], 'b.')
    # plt.show()
    #
    plt.plot(left_data['time'], left_data['Vx'], 'b-')
    plt.plot(steps['time'], steps['initVx'], 'r*')
    plt.xlabel('time')
    plt.ylabel('Vx(m/s)')
    plt.savefig(label + ' Vx')
    plt.clf()

    return steps


def main():
    right = GetData('../dataset/lzy_right_2017-12-04_17-17-01_-0800.csv')
    right = DealWithTime(right)
    right['accY'] *= 9.81
    right['accX'] *= 9.81
    right['accZ'] *= 9.81

    right = right[right['time'] > datetime.datetime(2017, 12, 5, 1, 17, 20, 000000)]
    right = right[right['time'] < datetime.datetime(2017, 12, 5, 1, 17, 39, 500000)]

    right = right.reset_index(drop=True)
    steps_right = process_data(right)

    global label
    label = 'Zhaoyang\'s left'
    left = GetData('../dataset/lzy_left_2017-12-04_17-16-52_-0800.csv')
    left = DealWithTime(left)
    left['accX'] *= 9.81
    left['accY'] *= 9.81
    left['accZ'] *= 9.81

    left = left[left['time'] > datetime.datetime(2017, 12, 5, 1, 17, 20, 000000 )]
    left = left[left['time'] < datetime.datetime(2017, 12, 5, 1, 17, 39, 500000 )]
    left = left.reset_index(drop=True)
    steps_left = process_data(left)
    
    steps_right = steps_right[0:-1]
    print('X displacement test')
    print(stats.normaltest(steps_left['calcX']).pvalue)
    print(stats.normaltest(steps_right['calcX']).pvalue)
    print(stats.ttest_ind(steps_left['calcX'], steps_right['calcX']).pvalue)
    print('')

    print('time test')
    print(stats.normaltest(steps_left['deltaT']).pvalue)
    print(stats.normaltest(steps_right['deltaT']).pvalue)
    print(stats.ttest_ind(steps_left['deltaT'], steps_right['deltaT']).pvalue)
    print(steps_left['deltaT'])
    print(steps_right['deltaT'])
    print('')

    print(stats.mannwhitneyu(steps_left['calcX'], steps_right['calcX']).pvalue)
    print(stats.mannwhitneyu(steps_left['deltaT'], steps_right['deltaT']).pvalue)
    print('walking speed left: ', end=' ')
    print(steps_left['calcX'].sum() / steps_left['deltaT'].sum())
    print('walking speed right: ', end=' ')
    print(steps_right['calcX'].sum() / steps_right['deltaT'].sum())
    # plt.plot(left['accZ'])
    # plt.show()

if __name__ == '__main__':
    main()
