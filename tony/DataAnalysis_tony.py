import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats

# This function get the data from CSV file
def GetData(filename):
    data = pd.read_csv(filename)
    data = data[['loggingTime(txt)','motionUserAccelerationX(G)', 'motionUserAccelerationY(G)', 'motionUserAccelerationZ(G)',\
                 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)',\
                 'motionYaw(rad)', 'motionRoll(rad)', 'motionPitch(rad)']]
    colums_name = ['timestamp','accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'yaw','roll', 'pitch']
    data.columns = colums_name
    return data

# This function convert the timestamp to pandas datatime object
def DealWithTime(data):
    data['time'] = pd.to_datetime(data['timestamp'])
    return data

# The gravity is 9.81 m/s^2 convert the Acceleration data to m/s^2
g = 9.81
def ScaleAcc(data):
    data['accX'] = data['accX']*g
    data['accY'] = data['accY']*g
    data['accZ'] = data['accZ']*g
    return data

# This function plot the Overview of the data points, it give brief introduction of the points shape
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

# fs means the frequency of the sensor, 30 data points pre second
fs = 30
# Fast Fourier Transform functions
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

# This function take the data points and return the Discrete Fourier Transform sample frequencies
def getStepFrequency(data):
    ffted = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(ffted))
    idx = np.argmax(np.abs(ffted))
    freq = freqs[idx]
    return abs(freq*fs)    

# This function separate each step and find the Mid-Stance
def seperateEachStep(data, step_cycle):
    steps = pd.DataFrame()
    idx_list = []
    for i in range(0, data.shape[0], step_cycle):
        idx = np.argmax(data['gyroZ'].iloc[i:i+step_cycle])
        if(idx != data.shape[0] - 1):
            idx_list.append(idx)
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

# These three function calculate the Speed from X,Y,Z direction
def calcSpeedY(data, begin, end):
    return data.loc[begin:end, 'accY'].sum()/fs
def calcSpeedZ(data, begin, end):
    return data.loc[begin:end, 'accZ'].sum()/fs
def calcSpeedX(data, begin, end):
    return data.loc[begin:end, 'accX'].sum()/fs

# This function return the corrected velocity of X and Y axis
def CorrectedV(begin, end, deltaVy, deltaVx, deltaT, initVy, initVx, data):
    data.set_value(begin, 'Vy', initVy)
    data.set_value(begin, 'Vx', initVx)
    for i in range(int(begin+1), int(end)):
        data.set_value(i,'Vy', data.loc[i-1,'Vy']+ (data.loc[i-1,'accY'] + deltaVy/deltaT)/fs)
        data.set_value(i,'Vx', data.loc[i-1,'Vx']+ (data.loc[i-1,'accX'] + deltaVx/deltaT)/fs)

# These functions return the distance of X and Y axis
def calcY(data, begin, end):
    return data.loc[begin:end, 'Vy'].sum() / fs
def calcX(data, begin, end):
    return data.loc[begin:end, 'Vx'].sum() / fs

# This function return the corrected velocity of Y axis
def CorrectedY(begin, end, calcY, deltaT, data):
    data.set_value(begin,'correctY', 0)
    for i in range(int(begin+1), int(end)):
        data.set_value(i,'correctY', data.loc[i-1,'correctY']+ (data.loc[i-1,'Vy'])/fs)
        
# getFC return the Foot Clearance result
def getFC(data, begin, end):
    idx = np.argmin(data.loc[begin:end, 'correctY'])
    return data.loc[idx, 'correctY']  

# This function plot after separate step result of GyroZ accX and accY
def plotSeperateStep(data, steps):
    plt.figure(figsize=(8,4))	 
    plt.plot(steps['gyroZ'],'r*', data['gyroZ'], 'b--')
    plt.title('step seperating according to gyroZ')
    plt.ylabel('gyroZ (rad/s)')
    plt.savefig(label + ' step seperating according to gyroZ')
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

# This function plot after FFT filtering Accelerometer data
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

# This function plot after FFT filtering Gyro data
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

# This function processing the data and provide the result we needed
# need two parameters
# 1. data set 
# 2. the sensor to the ankle length (meter)
def process_data(data, sensor_to_ankle_length):
    # Do Butterworth filter on Accelerometer data highcut = 5
    highcut = 5
    plotAboutFilteringAcc(data, highcut)
    data['accX'] = butter_lowpass_filter(data['accX'],highcut, fs)
    data['accY'] = butter_lowpass_filter(data['accY'],highcut, fs)
    data['accZ'] = butter_lowpass_filter(data['accZ'],highcut, fs)

    # Do Butterworth filter on gyro data highcut = 4.5, lowcut = 0.1
    lowcut = 0.1
    highcut = 4.5
    plotAboutFilteringGyro(data, highcut, lowcut)
    data['gyroX'] = butter_bandpass_filter(data['gyroX'],lowcut, highcut, fs)
    data['gyroY'] = butter_bandpass_filter(data['gyroY'],lowcut, highcut, fs)
    data['gyroZ'] = butter_bandpass_filter(data['gyroZ'],lowcut, highcut, fs)

    step_frequency = getStepFrequency(data['gyroZ'])
    print(label+' step frequency: ', end=' ')
    print(step_frequency)
    
    step_cycle = int(round(fs/step_frequency))
    steps = seperateEachStep(data, step_cycle)

    plotSeperateStep(data, steps)

    steps['initVy'] = 0
    steps['initVx'] = sensor_to_ankle_length * steps['gyroZ']
    steps['endVy'] = steps['initVy'].shift(-1)
    steps['endVx'] = steps['initVx'].shift(-1)

    steps['begin_idx'] = steps.index
    steps['end_idx'] = steps['begin_idx'].shift(-1)
    steps = steps.dropna()
    steps['end_idx'] = steps['end_idx'].astype(int)

    steps['calcVy'] = steps.apply((lambda row: calcSpeedY(data, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcVz'] = steps.apply((lambda row: calcSpeedZ(data, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcVx'] = steps.apply((lambda row: calcSpeedX(data, row['begin_idx'], row['end_idx'])), axis=1)

    steps['deltaVy'] = steps['endVy'] - steps['calcVy']
    steps['deltaVx'] = steps['endVx'] - steps['calcVx'] 
    steps['deltaT'] = (steps['end_idx'] - steps['begin_idx'])/fs
    steps = steps.reset_index(drop=True)

    begin = steps['begin_idx'].iloc[0]
    end = steps['end_idx'].iloc[-1]
    step_data = data[int(begin): int(end)]

    steps.apply((lambda row: CorrectedV(row['begin_idx'], row['end_idx'], row['deltaVy'], row['deltaVx'], row['deltaT'], row['initVy'], row['initVx'], step_data)), axis=1)

    steps['calcY'] = steps.apply((lambda row: calcY(step_data, row['begin_idx'], row['end_idx'])), axis=1)
    steps['calcX'] = steps.apply((lambda row: calcX(step_data, row['begin_idx'], row['end_idx'])), axis=1)

    steps.apply((lambda row: CorrectedY(row['begin_idx'], row['end_idx'], row['calcY'], row['deltaT'],step_data)), axis=1)

    steps['FC'] = steps.apply((lambda row: getFC(step_data, row['begin_idx'], row['end_idx'])), axis=1)

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

# In main function, we take both left and right feet data and do Data Analysis on them.
# It will print the result on screen and plot the figure we need to write the report.
def main():
    pd.options.mode.chained_assignment = None
    
    global label
    label = 'Tony\'s left'
    origin_left = GetData("../dataset/tony_left_2017-12-05_14-21-07_-0800.csv")
    origin_left = DealWithTime(origin_left)
    origin_left = ScaleAcc(origin_left)
#   By looking the data points on left foot, wo choose the reasonable middle data points 
    left = origin_left[700:1300]
    left = left.reset_index(drop=True)
#   The sensor to the ankle length measure result 15cm
    Lsteps = process_data(left, 0.15)

    label = 'Tony\'s right'
    origin_right =  GetData("../dataset/tony_right_2017-12-05_14-20-49_-0800.csv")
    origin_right = DealWithTime(origin_right)
    origin_right = ScaleAcc(origin_right)
#   By looking the data points on right foot, wo choose the reasonable middle data points 
    right = origin_right[1280:1850]
    right = right.reset_index(drop=True)
#   The sensor to the ankle length measure result 25cm
    Rsteps = process_data(right, 0.25)

    print("\nstep length (meter):")
    print("\nleft:")
    print(Lsteps['calcX'])
    print("right:")
    print(Rsteps['calcX'])

    print("\ntime of each step (second):")
    print("\nleft:")
    print(Lsteps['deltaT'])
    print("right:")
    print(Rsteps['deltaT'])
    
    print('\nstep length t-test')
    print("Left normaltest p: ", stats.normaltest(Lsteps['calcX']).pvalue)
    print("Right normaltest p: ", stats.normaltest(Rsteps['calcX']).pvalue)
    print("T test p: ", stats.ttest_ind(Lsteps['calcX'], Rsteps['calcX']).pvalue)

    print('\ntime t-test')
    print("Left normaltest p: ", stats.normaltest(Lsteps['deltaT']).pvalue)
    print("Right normaltest p: ", stats.normaltest(Rsteps['deltaT']).pvalue)
    print("T test p: ", stats.ttest_ind(Lsteps['deltaT'], Rsteps['deltaT']).pvalue)

    print('\nstep-length u-test p:')
    print(stats.mannwhitneyu(Lsteps['calcX'], Rsteps['calcX']).pvalue)
    print('\ntime u-test p:')
    print(stats.mannwhitneyu(Lsteps['deltaT'], Rsteps['deltaT']).pvalue)

    avg_Vx_left = Lsteps['calcX'].sum()/ Lsteps['deltaT'].sum()
    avg_Vx_right = Rsteps['calcX'].sum() / Rsteps['deltaT'].sum()
    print('\naverage walking speed left (m/s): ', end=' ')
    print(avg_Vx_left)
    print('average walking speed right (m/s): ', end=' ')
    print(avg_Vx_right)
    
    print('actual walking speed (m/s): ', end=' ')
    # Actual length measure by hand 20m
    actual_length = 20.0
    # 600 data points with frequency 30HZ
    actual_time = 600.0/fs
    actual_V = actual_length/actual_time
    print(actual_V)

    print('\naccuracy based on the average speed:')
    print('left: ', end=' ')
    print('{:.2%}'.format(abs(avg_Vx_left/actual_V)))
    print('right: ', end=' ')
    print('{:.2%}'.format(abs(avg_Vx_right/actual_V)))

    Rsteps.to_csv("../dataset/tony_steps_right.csv")
    Lsteps.to_csv("../dataset/tony_steps_left.csv")

if __name__ == '__main__':
    main()