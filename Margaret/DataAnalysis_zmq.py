import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import re
from scipy import stats

# read in the data, change the name of columns
def GetData(filename):
    data = pd.read_csv(filename)
    data = data[['loggingTime(txt)','motionUserAccelerationX(G)', 'motionUserAccelerationY(G)', 'motionUserAccelerationZ(G)',\
                 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)',\
                 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']]
    colums_name = ['timestamp','accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'yaw','roll', 'pitch']
    data.columns = colums_name
    return data

# convert the timestamp column
def DealWithTime(data):
    data['time'] = pd.to_datetime(data['timestamp'])
    return data

# the local gravity (m/s^-2)
g = 9.81 

# convert the accelerate data from G to m/s
def ScaleAcc(data):
    data['accX'] = data['accX']*g
    data['accY'] = data['accY']*g
    data['accZ'] = data['accZ']*g
    return data

# plot all the data to have an overview
def Overview(data):
    plt.figure(figsize=(15,20))
    p1 = plt.subplot(3,1,1)
    p2 = plt.subplot(3,1,2)
    p3 = plt.subplot(3,1,3)
    p1.plot(data['accX'])
    p2.plot(data['accY'])
    p3.plot(data['accZ'])
    plt.show()

    plt.figure(figsize=(15,20))
    p4 = plt.subplot(3,1,1)
    p5 = plt.subplot(3,1,2)
    p6 = plt.subplot(3,1,3)
    p4.plot(data['gyroX'])
    p5.plot(data['gyroY'])
    p6.plot(data['gyroZ'])
    plt.show()

# the frquency of sampling (Hz)
fs = 30

# butterworth filter functions
# reference:
# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
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

# get step frequency using fourier transformation
# the step frequency should be the peak frequency after FFT 
# reference: 
# https://stackoverflow.com/questions/3694918/how-to-extract-frequency-associated-with-fft-values-in-python 
def getStepFrequency(data):
    ffted = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(ffted))
    idx = np.argmax(np.abs(ffted))
    freq = freqs[idx]
    return abs(freq*fs) 

# seperate the continous walking data into a series of stride cycles
# use the mid-stance point for segmentation
def seperateEachStep(data, step_cycle):
    steps = pd.DataFrame()
    idx_list = []
    for i in range(0, data.shape[0], step_cycle):
        idx = np.argmax(data['gyroZ'].iloc[i:i+step_cycle])
        if(idx != data.shape[0] - 1):
            idx_list.append(idx)
    # print(idx_list)
    for i in range(0, len(idx_list)-1):
        prev = idx_list[i]
        nxt = idx_list[i+1]
        while (data['gyroZ'].iloc[prev]>data['gyroZ'].iloc[prev+1]):
            prev +=1
        while (data['gyroZ'].iloc[nxt]> data['gyroZ'].iloc[nxt-1]):
            nxt -=1
        floor1 = prev;
        floor2 = nxt;
        if(floor1 < floor2):
            idx = np.argmax(data['gyroZ'].iloc[floor1:floor2])
            steps = steps.append(data.iloc[idx])
    return steps

# calculate the speed for each step (without correction)
def calcSpeedY(data, begin, end):
    return data.loc[begin:end, 'accY'].sum()/fs
def calcSpeedZ(data, begin, end):
    return data.loc[begin:end, 'accZ'].sum()/fs
def calcSpeedX(data, begin, end):
    return data.loc[begin:end, 'accX'].sum()/fs

# calculate and correct Vx and Vy for each time point
def CorrectedV(begin, end, deltaVy, deltaVx, deltaT, initVy, initVx, data):
    data.set_value(begin, 'Vy', initVy)
    data.set_value(begin, 'Vx', initVx)
    for i in range(int(begin+1), int(end)):
        data.set_value(i,'Vy', data.loc[i-1,'Vy']+ (data.loc[i-1,'accY'] + deltaVy/deltaT)/fs)
        data.set_value(i,'Vx', data.loc[i-1,'Vx']+ (data.loc[i-1,'accX'] + deltaVx/deltaT)/fs)

# calculate the distance for each step
def calcY(data, begin, end):
    return data.loc[begin:end, 'Vy'].sum() / fs
def calcX(data, begin, end):
    return data.loc[begin:end, 'Vx'].sum() / fs

# correct distance of the Y axis for each time point
def CorrectedY(begin, end, calcY, deltaT, data):
    data.set_value(begin,'correctY', 0)
    for i in range(int(begin+1), int(end)):
        data.set_value(i,'correctY', data.loc[i-1,'correctY']+ (data.loc[i-1,'Vy'])/fs)
 
# calculate the Foot Clearance
def getFC(data, begin, end):
    idx = np.argmin(data.loc[begin:end, 'correctY'])
    return data.loc[idx, 'correctY']  

# plot some figure for checking the seperating result
def plotSeperateStep(data, steps):
    plt.figure(figsize=(8,4))    
    plt.plot(steps['gyroZ'],'r*', data['gyroZ'], 'b--')
    plt.title('step seperating according to gyroZ')
    plt.ylabel('gyroZ (rad/s)')
    plt.savefig('step seperating according to gyroZ')
    plt.clf()
    
    plt.plot(steps['accY'],'r*', data['accY'], 'b--')
    plt.title(label + ' accY after step seperating')
    plt.ylabel('accY ( m/(s^2) )')
    plt.savefig(label + ' accY after step seperating')
    plt.clf()
    
    plt.plot(steps['accX'],'r*', data['accX'], 'b--')
    plt.title(label + ' accX after step seperating')
    plt.ylabel('accX ( m/(s^2) )')
    plt.savefig(label + ' accX after step seperating')
    plt.clf()

# plot some figure for checking the filtering result
def plotAboutFilteringAcc(data, highcut):
    plt.figure(figsize=(10,8))
    filtered = butter_lowpass_filter(data['accX'],highcut, fs)
    p1 = plt.subplot(2,1,1)
    p2 = plt.subplot(2,1,2)
    p1.plot(data['accX'])
    p1.set_title(label + ' accX before filtering')
    p1.set_ylabel('accX ( m/(s^2) )')
    p2.plot(filtered)
    p2.set_title(label + ' accX after filtering')
    p2.set_ylabel('accX ( m/(s^2) )')
    plt.savefig(label + ' accX before & after filtering')
    plt.clf()

    filtered = butter_lowpass_filter(data['accY'],highcut, fs)
    p1 = plt.subplot(2,1,1)
    p2 = plt.subplot(2,1,2)
    p1.plot(data['accY'])
    p1.set_title(label + ' accY before filtering')
    p1.set_ylabel('accY ( m/(s^2) )')
    p2.plot(filtered)
    p2.set_title(label + ' accY after filtering')
    p2.set_ylabel('accY ( m/(s^2) )')
    plt.savefig(label + ' accY before & after filtering')
    plt.clf()

    filtered = butter_lowpass_filter(data['accZ'],highcut, fs)
    p1 = plt.subplot(2,1,1)
    p2 = plt.subplot(2,1,2)
    p1.plot(data['accZ'])
    p1.set_title(label + ' accZ before filtering')
    p1.set_ylabel('accZ ( m/(s^2) )')
    p2.plot(filtered)
    p2.set_title(label + ' accZ after filtering')
    p2.set_ylabel('accZ ( m/(s^2) )')
    plt.savefig(label + ' accZ before & after filtering')
    plt.clf()

def plotAboutFilteringGyro(data, highcut, lowcut):  
    filtered = butter_bandpass_filter(data['gyroZ'],lowcut, highcut, fs)
    p1 = plt.subplot(2,1,1)
    p2 = plt.subplot(2,1,2)
    p1.plot(data['gyroZ'])
    p1.set_title(label + ' gyroZ before filtering')
    p1.set_ylabel('gyroZ (rad/s)')
    p2.plot(filtered)
    p2.set_title(label + ' gyroZ after filtering')
    p2.set_ylabel('gyroZ (rad/s)')
    plt.savefig(label + ' gyroZ before & after filtering')
    plt.clf()

    filtered = butter_bandpass_filter(data['gyroX'],lowcut, highcut, fs)
    p1 = plt.subplot(2,1,1)
    p2 = plt.subplot(2,1,2)
    p1.plot(data['gyroX'])
    p1.set_title(label + ' gyroX before filtering')
    p1.set_ylabel('gyroX (rad/s)')
    p2.plot(filtered)
    p2.set_title(label + ' gyroX after filtering')
    p2.set_ylabel('gyroX (rad/s)')
    plt.savefig(label + ' gyroX before & after filtering')
    plt.clf()

    filtered = butter_bandpass_filter(data['gyroY'],lowcut, highcut, fs)
    p1 = plt.subplot(2,1,1)
    p2 = plt.subplot(2,1,2)
    p1.plot(data['gyroY'])
    p1.set_title(label + ' gyroY before filtering')
    p1.set_ylabel('gyroY (rad/s)')
    p2.plot(filtered)
    p2.set_title(label + ' gyroY after filtering')
    p2.set_ylabel('gyroY (rad/s)')
    plt.savefig(label + ' gyroY before & after filtering')
    plt.clf()

# the data processing pipeline
label = 'Manqing\'s right'
def process_data(data, sensor_to_ankle):

    # noise filtering by butterworth filter
    highcut = 5
    plotAboutFilteringAcc(data, highcut)
    data['accX'] = butter_lowpass_filter(data['accX'],highcut, fs)
    data['accY'] = butter_lowpass_filter(data['accY'],highcut, fs)
    data['accZ'] = butter_lowpass_filter(data['accZ'],highcut, fs)

    lowcut = 0.1
    highcut = 4.5
    plotAboutFilteringGyro(data, highcut, lowcut)
    data['gyroX'] = butter_bandpass_filter(data['gyroX'],lowcut, highcut, fs)
    data['gyroY'] = butter_bandpass_filter(data['gyroY'],lowcut, highcut, fs)
    data['gyroZ'] = butter_bandpass_filter(data['gyroZ'],lowcut, highcut, fs)

    # get step frequency
    step_frequency = getStepFrequency(data['gyroZ'])
    print(label+' walking pace: ', end=' ')
    print(str(step_frequency*60) + ' steps/minute')
    
    # seperate the data to get each single step
    step_cycle = int(round(fs/step_frequency))
    steps = seperateEachStep(data, step_cycle)

    plotSeperateStep(data, steps)

    # record the begin velocity and the end velocity of each step
    # sensor_to_ankle is the distance between the phone and the ankle joint
    steps['initVy'] = 0
    steps['initVx'] = sensor_to_ankle * steps['gyroZ']
    steps['endVy'] = steps['initVy'].shift(-1)
    steps['endVx'] = steps['initVx'].shift(-1)

    # record the begin point and the end point of each step
    steps['begin_idx'] = steps.index
    steps['end_idx'] = steps['begin_idx'].shift(-1)
    steps = steps.dropna()
    steps['end_idx'] = steps['end_idx'].astype(int)

    # calculate the speed for each step (without correction)
    steps['calcVy'] = steps.apply((lambda row: calcSpeedY(data, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcVz'] = steps.apply((lambda row: calcSpeedZ(data, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcVx'] = steps.apply((lambda row: calcSpeedX(data, row['begin_idx'], row['end_idx'])), axis=1)

    # calculte the coefficient used to correct speed
    steps['deltaVy'] = steps['endVy'] - steps['calcVy']
    steps['deltaVx'] = steps['endVx'] - steps['calcVx'] 
    steps['deltaT'] = (steps['end_idx'] - steps['begin_idx'])/fs
    steps = steps.reset_index(drop=True)

    # slice the data to make sure it only contains the valid steps
    begin = steps['begin_idx'].iloc[0]
    end = steps['end_idx'].iloc[-1]
    step_data = data[int(begin): int(end)]

    # calculate and correct the speed for each time point
    steps.apply((lambda row: CorrectedV(row['begin_idx'], row['end_idx'], row['deltaVy'], row['deltaVx'], row['deltaT'],\
                                   row['initVy'], row['initVx'], step_data)), axis=1)

    # calculte the distance for each step
    steps['calcY'] = steps.apply((lambda row: calcY(step_data, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcX'] = steps.apply((lambda row: calcX(step_data, row['begin_idx'], row['end_idx'])), axis=1)

    # calculate the distance of the Y axis for each time point
    steps.apply((lambda row: CorrectedY(row['begin_idx'], row['end_idx'], row['calcY'], row['deltaT'],step_data)), axis=1)

    # calculate the foor clearance
    steps['FC'] = steps.apply((lambda row: getFC(step_data, row['begin_idx'], row['end_idx'])), axis=1)

    # show the Vy and Vx for each time point
    plt.plot(step_data['time'], step_data['Vy'], 'b-')
    plt.plot(steps['time'], steps['initVy'], 'r*')
    plt.xlabel('time')
    plt.ylabel('Vy(m/s)')
    plt.title(label + ' Vy')
    plt.savefig(label + ' Vy')
    plt.clf()

    plt.plot(step_data['time'], step_data['Vx'], 'b-')
    plt.plot(steps['time'], steps['initVx'], 'r*')
    plt.xlabel('time')
    plt.ylabel('Vx(m/s)')
    plt.title(label + ' Vx')
    plt.savefig(label + ' Vx')
    plt.clf()

    return steps

def main():
    pd.options.mode.chained_assignment = None

    # read data for right foot
    origin_right =  GetData("../dataset/zmq_right_2017-12-04_17-02-36_-0800.csv")

    # convert the timestamp format
    origin_right = DealWithTime(origin_right)

    # convert the acceleration data
    origin_right = ScaleAcc(origin_right)

    # slice to get the part of valid walking data
    # Overview(origin_left[500:1350])
    right = origin_right[620:1400]
    right = right.reset_index(drop=True)

    # process data
    Rsteps = process_data(right, 0.13)

    # do same thing for left foot
    global label
    label = 'Manqing\'s left'

    origin_left = GetData("../dataset/zmq_left_2017-12-04_17-02-37_-0800.csv")
    
    origin_left = DealWithTime(origin_left)
    origin_left = ScaleAcc(origin_left)

    left = origin_left[600:1300]
    left = left.reset_index(drop=True)
    
    Lsteps = process_data(left, 0.11)

    # slice the step_data for left and right foot to make sure they can match
    Rsteps = Rsteps[1:-1]

    # print the result of step length and time for each step
    print("\nstep length:")
    print("right:")
    print(Rsteps['calcX'])
    print("\nleft:")
    print(Lsteps['calcX'])
    print("\ntime of each step:")
    print("right:")
    print(Rsteps['deltaT'])
    print("\nleft:")
    print(Lsteps['deltaT'])

    # test whether the gait is symmetric
    print('step length normal test')
    print('right: ', end=' ')
    print(stats.normaltest(Rsteps['calcX']).pvalue)
    print('left: ', end=' ')
    print(stats.normaltest(Lsteps['calcX']).pvalue)
    print('\nstep length t-test')
    print(stats.ttest_ind(Lsteps['calcX'], Rsteps['calcX']).pvalue)

    print('time normal test')   
    print('right: ', end=' ')
    print(stats.normaltest(Rsteps['deltaT']).pvalue)    
    print('left: ', end=' ')
    print(stats.normaltest(Lsteps['deltaT']).pvalue)
    print('\ntime t-test')
    print(stats.ttest_ind(Lsteps['deltaT'], Rsteps['deltaT']).pvalue)

    print('\nstep-length u-test')
    print(stats.mannwhitneyu(Lsteps['calcX'], Rsteps['calcX']).pvalue)
    print('\ntime u-test')
    print(stats.mannwhitneyu(Lsteps['deltaT'], Rsteps['deltaT']).pvalue)
    
    # test the accuarcy of the result using the average speed
    avg_Vx_right = Rsteps['calcX'].sum()/ Rsteps['deltaT'].sum()
    avg_Vx_left = Lsteps['calcX'].sum()/ Lsteps['deltaT'].sum()
    
    print('\naverage walking speed right: ', end=' ')
    print(avg_Vx_right)
    print('average walking speed left: ', end=' ')
    print(avg_Vx_left)

    actual_length = 21.0
    actual_time = 820.0/fs  
    actual_V = actual_length/actual_time
    print('actual walking speed: ', end=' ')
    print(actual_V)

    print('\naccuracy based on the average speed:')
    print('right: ', end=' ')
    print('{:.2%}'.format(abs(avg_Vx_right/actual_V)))
    print('left: ', end=' ')
    print('{:.2%}'.format(abs(avg_Vx_left/actual_V)))

    Rsteps.to_csv("../dataset/zmq_steps_right.csv")
    Lsteps.to_csv("../dataset/zmq_steps_left.csv")

if __name__ == '__main__':
    main()
