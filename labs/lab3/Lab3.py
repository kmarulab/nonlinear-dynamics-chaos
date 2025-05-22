import pandas as pd
from statistics import mean,median,stdev
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def linear_fit(v,m,c):
    return m*v +c

def getData(file):
    tab1 = pd.read_csv(file,
                    index_col=None,
                    header=None,
                    names=['Time','Pendulum_V'],
                    skiprows=9)
    if(tab1['Time'][0] != 0 and tab1['Time'][0]<0):                
        tab1['Time'] = tab1['Time'] + abs(tab1['Time'][0])
    elif(tab1['Time'][0] != 0 and tab1['Time'][0]>0):
        tab1['Time'] = tab1['Time'] - abs(tab1['Time'][0])

    return tab1

degrees= np.array([0,10,30,60,90,120,150, 170,190,210,240,270,300,330,350])
arr = np.array([])
for i in range(0,15,1):
    file_name = "lab3Data2/calibration/lower_pendulum_callibration"+str(i)+".csv"
    data = getData(file_name) 
    v_data = data["Pendulum_V"][:25000]
    avg = np.mean(v_data)
    arr = np.append(arr,avg)

plt.scatter(arr, degrees, label='Manual Calibration')
plt.xlabel('Voltage')
plt.ylabel('Degrees')
plt.title('Lower Pendulum Raw Calibration Data')
plt.legend()
plt.show()

def off_Set_deg(arr, deg_amt):
    new_arr = arr - deg_amt
    new_arr = np.where(new_arr < 0, new_arr + 360, new_arr)
    return new_arr
degrees_offset = off_Set_deg(degrees,150)

shift_by = np.where(degrees_offset == 0)[0][0]
shift_arr = np.roll(arr,-shift_by)
plt.scatter(shift_arr,degrees,label='adjusted offset')

linearfit = curve_fit(linear_fit,shift_arr,degrees)
m = linearfit[0][0]
n = linearfit[0][1]
coef_lower = [m,n]
voltage = np.linspace(0.,0.063,100)
linearfit_lower = linear_fit(voltage,m,n)
plt.plot(voltage, linearfit_lower,label='linear fit')
plt.xlabel('Voltage')
plt.ylabel('Degrees')
plt.legend()
plt.title('Lower Pendulum Calibration')
plt.show()

degrees= np.array([0,10,30,60,90,120,150, 170,190,210,240,270,300,330,350])
arr = np.array([])
for i in range(1,16,1):
    file_name = "lab3Data2/calibration/upper_pendulum_callibration"+str(i)+".csv"
    data = getData(file_name) 
    v_data = data["Pendulum_V"][:25000]
    avg = np.mean(v_data)
    arr = np.append(arr,avg)

plt.scatter(arr, degrees, label='Manual Calibration')
plt.xlabel('Voltage')
plt.ylabel('Degrees')
plt.title('Upper Pendulum Raw Calibration Data')
plt.legend()
plt.show()

def off_Set_deg(arr, deg_amt):
    new_arr = arr - deg_amt
    new_arr = np.where(new_arr < 0, new_arr + 360, new_arr)
    return new_arr
degrees_offset = off_Set_deg(degrees,170)

shift_by = np.where(degrees_offset == 0)[0][0]
shift_arr = np.roll(arr,-shift_by)
plt.scatter(shift_arr,degrees,label='adjusted offset')

linearfit = curve_fit(linear_fit,shift_arr,degrees)
m = linearfit[0][0]
n = linearfit[0][1]
coef_upper = [m,n]
voltage = np.linspace(0.,0.063,100)
linearfit_lower = linear_fit(voltage,m,n)
plt.plot(voltage, linearfit_lower,label='linear fit')
plt.xlabel('Voltage')
plt.ylabel('Degrees')
plt.legend()
plt.title('Upper Pendulum Calibration')
plt.show()

def getExperimentData(file):
    tab1 = pd.read_csv(file,
                    index_col=None,
                    header=None,
                    names=['Time','LowerPendulum_V','UpperPendulum_V'],
                    skiprows=11)
    if(tab1['Time'][0] != 0 and tab1['Time'][0]<0):                
        tab1['Time'] = tab1['Time'] + abs(tab1['Time'][0])
    elif(tab1['Time'][0] != 0 and tab1['Time'][0]>0):
        tab1['Time'] = tab1['Time'] - abs(tab1['Time'][0])

    return tab1

def angles_data_elementwise(v_data, m, n):
    out = linear_fit(v_data, m,n)
    return out


def getAngles(file):
    repeatability = getExperimentData(file)
    t_data = np.array(repeatability["Time"][:])
    v_data_l = np.array(repeatability["LowerPendulum_V"][:])
    angles_data_l = angles_data_elementwise(v_data_l, *coef_lower)
    v_data_u = np.array(repeatability["UpperPendulum_V"][:])
    angles_data_u = angles_data_elementwise(v_data_u, *coef_upper)

    return t_data,angles_data_l, angles_data_u

repeatability1 = getAngles('lab3Data2/repeatability/repeatability1.csv')
repeatability2 = getAngles('lab3Data2/repeatability/repeatability2.csv')
repeatability3 = getAngles('lab3Data2/repeatability/repeatability3.csv')
repeatability4 = getAngles('lab3Data2/repeatability/repeatability4.csv')
repeatability5 = getAngles('lab3Data2/repeatability/repeatability5.csv')

plt.plot(repeatability1[0], repeatability1[1], label = 'exp 1 lower pendulum')
plt.plot(repeatability1[0],repeatability1[2], label = 'exp 1 upper pendulum ')

plt.plot(repeatability2[0], repeatability2[1], label = 'exp 2 lower pendulum')
plt.plot(repeatability2[0],repeatability2[2], label = 'exp 2 upper pendulum ')

plt.plot(repeatability3[0], repeatability3[1], label = 'exp 3 lower pendulum')
plt.plot(repeatability3[0],repeatability3[2], label = 'exp 3 upper pendulum ')

plt.plot(repeatability4[0], repeatability4[1], label = 'exp 4 lower pendulum')
plt.plot(repeatability4[0],repeatability4[2], label = 'exp 4 upper pendulum ')

plt.plot(repeatability5[0], repeatability5[1], label = 'exp 1 lower pendulum')
plt.plot(repeatability5[0],repeatability5[2], label = 'exp 1 upper pendulum ')

plt.legend()
plt.title('Repeatability Experiment Angles vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')

plt.show()

plt.plot(repeatability1[1], repeatability1[2], label = 'exp 1 pendulum')
plt.plot(repeatability2[1], repeatability2[2], label = 'exp 2 pendulum ')
plt.plot(repeatability3[1], repeatability3[2], label = 'exp 3 pendulum')
plt.plot(repeatability4[1], repeatability4[2], label = 'exp 4 pendulum ')
plt.plot(repeatability5[1], repeatability5[2], label = 'exp 5 pendulum')
plt.title('Phase Plot')
plt.xlabel('Lower Pendulum Angle (deg)')
plt.ylabel('Upper Pendulum Angle (deg')
plt.legend()
plt.show()

ang_velocity_1 = getAngles('lab3Data2/angular-velocity/angvel 01.csv')
angular_velocity_1_u =np.diff(ang_velocity_1[1]*np.pi/180)/np.diff(ang_velocity_1[0])
angular_velocity_1_l =np.diff( ang_velocity_1[2]*np.pi/180)/np.diff(ang_velocity_1[0])
angular_velocity_1_u = np.insert(angular_velocity_1_u, 0,0)
angular_velocity_1_l = np.insert(angular_velocity_1_l, 0,0)
plt.plot(ang_velocity_1[1]*np.pi/180, angular_velocity_1_u, label='upper pendulum')
plt.plot(ang_velocity_1[1]*np.pi/180, angular_velocity_1_l, label='lower pendulum')
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity(rad/s)')
plt.title('Angular Velocity Test 1')
plt.legend()
plt.show()

ang_velocity_2 = getAngles('lab3Data2/angular-velocity/angvel 02.csv')
angular_velocity_2_u =np.diff(ang_velocity_2[1]*np.pi/180)/np.diff(ang_velocity_2[0])
angular_velocity_2_l =np.diff( ang_velocity_2[2]*np.pi/180)/np.diff(ang_velocity_2[0])
angular_velocity_2_u = np.insert(angular_velocity_2_u, 0,0)
angular_velocity_2_l = np.insert(angular_velocity_2_l, 0,0)
plt.plot(ang_velocity_2[1]*np.pi/180, angular_velocity_2_u, label='upper pendulum')
plt.plot(ang_velocity_2[1]*np.pi/180, angular_velocity_2_l, label='lower pendulum')
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity(rad/s)')
plt.title('Angular Velocity Test 2')
plt.legend()
plt.show()

ang_velocity_3 = getAngles('lab3Data2/angular-velocity/angvel 03.csv')
angular_velocity_3_u =np.diff(ang_velocity_3[1]*np.pi/180)/np.diff(ang_velocity_3[0])
angular_velocity_3_l =np.diff( ang_velocity_3[2]*np.pi/180)/np.diff(ang_velocity_3[0])
angular_velocity_3_u = np.insert(angular_velocity_3_u, 0,0)
angular_velocity_3_l = np.insert(angular_velocity_3_l, 0,0)
plt.plot(ang_velocity_3[1]*np.pi/180, angular_velocity_3_u, label='upper pendulum')
plt.plot(ang_velocity_3[1]*np.pi/180, angular_velocity_3_l, label='lower pendulum')
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity(rad/s)')
plt.title('Angular Velocity Test 3')
plt.legend()
plt.show()

def count_rotations(angle_series, units="deg"):
    θ = np.asarray(angle_series, dtype=float)
    if units == "deg":
        θ = np.deg2rad(θ)

    θ_unwrapped = np.unwrap(θ)        
    dθ = np.diff(θ_unwrapped) 

    cw = ccw = 0
    partial = 0.0
    for delta in dθ:
        partial += delta
        while partial >= 2*np.pi:      
            ccw    += 1
            partial -= 2*np.pi
        while partial <= -2*np.pi:  
            cw     += 1
            partial += 2*np.pi

    return cw, ccw

predictability1_1 = getAngles('lab3Data2/predictability/predictability1_1.csv')
predictability1_2 = getAngles('lab3Data2/predictability/predictability1_2.csv')
predictability1_3 = getAngles('lab3Data2/predictability/predictability1_3.csv')
predictability2_1 = getAngles('lab3Data2/predictability/predictability2_1.csv')
predictability2_2 = getAngles('lab3Data2/predictability/predictability2_2.csv')
predictability2_3 = getAngles('lab3Data2/predictability/predictability2_3.csv')
predictability3_1 = getAngles('lab3Data2/predictability/predictability3_1.csv')
predictability3_2 = getAngles('lab3Data2/predictability/predictability3_2.csv')
predictability3_3 = getAngles('lab3Data2/predictability/predictability3_3.csv')

rotations1_1 = [count_rotations(predictability1_1[1],'deg'), count_rotations(predictability1_1[2],'deg')]
rotations1_2 = [count_rotations(predictability1_2[1],'deg'), count_rotations(predictability1_2[2],'deg')]
rotations1_3 = [count_rotations(predictability1_3[1],'deg'), count_rotations(predictability1_3[2],'deg')]
rotations2_1 = [count_rotations(predictability2_1[1],'deg'), count_rotations(predictability2_1[2],'deg')]
rotations2_2 = [count_rotations(predictability2_2[1],'deg'), count_rotations(predictability2_2[2],'deg')]
rotations2_3 = [count_rotations(predictability2_3[1],'deg'), count_rotations(predictability2_3[2],'deg')]
rotations3_1 = [count_rotations(predictability3_1[1],'deg'), count_rotations(predictability3_1[2],'deg')]
rotations3_2 = [count_rotations(predictability3_2[1],'deg'), count_rotations(predictability3_2[2],'deg')]
rotations3_3 = [count_rotations(predictability3_3[1],'deg'), count_rotations(predictability3_3[2],'deg')]

rotations = [[rotations1_1,rotations1_2,rotations1_3],
        [rotations2_1,rotations2_2,rotations2_3],
        [rotations3_1,rotations3_2,rotations3_3]]

def build_rotation_table(data):
    rows = []
    for exp_idx, experiment in enumerate(data, start=1):
        for trial_idx, run in enumerate(experiment, start=1):
            upper, lower = run        
            rows.append({
                "experiment": exp_idx,
                "trial":      trial_idx,
                "upper_cw":   upper[0],
                "upper_ccw":  upper[1],
                "lower_cw":   lower[0],
                "lower_ccw":  lower[1],
            })

    return pd.DataFrame(rows)

print(build_rotation_table(rotations))

