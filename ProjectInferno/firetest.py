import matplotlib.pyplot as plt
import numpy as np
from numpy.core.function_base import linspace
from numpy.lib.polynomial import poly, polyval
import pandas as pd
from scipy.optimize import curve_fit

#
#Loading curated data from csv into pandas dataframes
#

test6mm1 = pd.read_csv('6mmfiretest1.csv', sep = ';')
test6mm2 = pd.read_csv('6mmfiretest2.csv', sep = ';')
test6mm3 = pd.read_csv('6mmfiretest3.csv', sep = ';')
test6mm4 = pd.read_csv('6mmfiretest4.csv', sep = ';')
test6mm5 = pd.read_csv('6mmfiretest5.csv', sep = ';')
test2mm1 = pd.read_csv('2mmfiretest1.csv', sep = ';')
test10mm1 = pd.read_csv('10mmfiretest1.csv', sep = ';')
test10mm2 = pd.read_csv('10mmfiretest2.csv', sep = ';')
testrates = pd.read_csv('TestRates.csv', sep = ';')
Applicablerates = pd.read_csv('TestRates copy.csv', sep =';')
distanceTime6mm = pd.read_csv('distanceTime6mm.csv', sep = ';')
typej = pd.read_csv('TypeJ.csv', sep = ';')


#conversion of voltage to temperature based on a 2 degree polynomial fit of the typeJ conversion table
x = linspace(0, 360, num = 1000)
coefficients = np.polyfit(typej['mV'], typej['T'], 2)
#print(test2mm1)
def convertTotemp(channel):
    channelset = channel+1.174
    #print(channel)
    #print(channelset)
    fit = np.polyval(coefficients, channelset)
    #print(fit)
    return(fit)

#plot of type j table and fit
'''
fig, ax = plt.subplots()
ax.plot(typej['T'], typej['mV'])
ax.plot(x, fit)
plt.show()
'''

dist = linspace(0,300, num =1000)

'''
distanceTime10mm = np.array([
#   Thermocouple channel, distance (mm), time test1, time (test 2)
    [7,                   0,            136.0311087,          167.01699],
    [6,                   32,           167.020109,           228.997999756],
    [5,                   64,           217.003109,           361.982999998]
])

#plot of 10mm rates

rate10mm1 = np.polyfit(distanceTime10mm[:,2]-136.031187, distanceTime10mm[:,1],1)
rate10mm2 = np.polyfit(distanceTime10mm[:,3]- 167.01699, distanceTime10mm[:,1],1)
fig, axs = plt.subplots(2)
axs[0].plot(distanceTime10mm[:,2] -136.031187, distanceTime10mm[:,1],'o')
axs[0].plot(dist, np.polyval(rate10mm1, dist))
axs[0].set_title('Experiment 1')
axs[1].plot(distanceTime10mm[:,3] - 167.01699, distanceTime10mm[:,1],'o')
axs[1].plot(dist, np.polyval(rate10mm2, dist))
axs[1].set_title('Experiment 2 (not valid)')
for ax in axs.flat:
    ax.set(xlabel='Distance (mm)', ylabel='Time (seconds)')
for ax in axs.flat:
    ax.label_outer()

plt.show()
rates10mm = []
rates10mm.append(rate10mm1[0])
rates10mm.append(0)
print(rates10mm)
'''

#weber function for fit
def Weber(s, a):
    v = 1.346/(a*(s**2))
    return v

def vogel(s, a):
    l = a*s**3
    return l


experimentalData = np.array([
    #spacing, length
    [2, 60],
    [6, 60],
    [10, 60],
])

vogeldata = np.array([
    [6.53, 50.8, 0.759],
    [6.53, 40.64, .5534]
])

p, c = curve_fit(Weber, Applicablerates['Spacing'],Applicablerates['Velocity (mean)'])


#the scipy curve fit allows fitting to  a specified function in this case the weber model defined above
parameters, covariance = curve_fit(Weber, testrates['Spacing'], testrates['Velocity (mean)'])
print(parameters[0], p[0])


#Creating a series of values for plotting fits and finding the best polynomial fit for data
xval = linspace(1.5, 11, num =40)
bestFit = np.polyfit(testrates['Spacing'], testrates['Velocity (mean)'], 1)
'''
#Plotting found data, weber fit, and polynomial fit
fig, ax = plt.subplots()
ax.scatter(testrates['Spacing'], testrates['Velocity (mean)'], color = 'orange', label = '2mm Data')
ax.scatter(Applicablerates['Spacing'], Applicablerates['Velocity (mean)'], color = 'blue', label = 'Experimental Data')
ax.scatter(vogeldata[:,0], vogeldata[:,2], color = 'red', label = 'Vogel & Williams data')
#ax.plot(xval, np.polyval(bestFit, xval), color = 'orange', label = 'Polynomial fit')
ax.plot(xval, Weber(xval, parameters[0]), color = 'orange', label = 'Weber Model fit \'a\' value')
ax.plot(xval, Weber(xval, 0.119), color = 'red', label = 'Vogel & William\'s \'a\' value')
ax.plot(xval, Weber(xval, p[0]), color = 'blue', label = 'Weber Model not including 2mm')
plt.title('Comparison of Weber model based on fit \'a\' values')
plt.xlabel('Spacing (mm)')
plt.ylabel('Propagation rate (mm/s)')
plt.legend()
plt.show()
'''
weberVelocities = Weber(testrates['Spacing'], p[0])
error = (testrates['Velocity (mean)'] - weberVelocities)
print(error)

'''
differencesWeber = np.abs(Weber(xval, parameters[0], parameters[1]) - Weber(xval, p[0], 1.1))
plt.plot(xval, differencesWeber)
plt.xlabel('Spacing (mm)')
plt.ylabel('Differences')
plt.title('Deviation of Vogel fit from Weber fit')
plt.show()

sval = np.linspace(0,10,num=1000)
differencesVogel = vogel(sval, parameters[0]) - vogel(sval, p[0])
plt.plot(sval, differencesVogel)
plt.xlabel('Spacing (mm)')
plt.ylabel('Differences')
plt.title('Deviation of Weber fit from Vogel fit')
plt.show()


sval = np.linspace(0,11,num=1000)
fig, ax = plt.subplots()
ax.scatter(experimentalData[:,0], experimentalData[:,1], color = 'blue', label = 'experimental data')
ax.scatter(vogeldata[:,0], vogeldata[:,1], color = 'red', label = 'Vogel & Williams data')
ax.plot(sval, vogel(sval, parameters[0]), color = 'blue' , label = 'Weber Model fit \'a\' value')
ax.plot(sval, vogel(sval, .119), color = 'red', label = 'Vogel & William\'s \'a\' value')
ax.plot(sval, vogel(sval, p[0]), color = 'orange',label = 'Weber Model fit \'a\' value not including 2mm')
plt.fill_between(sval, vogel(sval, p[0]), 0, alpha = 0.3, color = 'red')
plt.fill_between(sval, vogel(sval, p[0]), vogel(sval, parameters[0]), alpha = 0.3, color = 'yellow')
plt.fill_between(sval, vogel(sval, parameters[0]), vogel(sval, parameters[0]).max(), alpha = 0.3, color = 'green')
plt.legend()
plt.xlabel('Spacing (mm)')
plt.ylabel('Height (mm)')
plt.title('Marginal Propagation Behavior comparison to model predictions')
plt.show()
#print(p[0], parameters[0])

'''
#plots of distance time with fits for 6mm tests

rate6mm1 = np.polyfit(distanceTime6mm['Experiment 1'], distanceTime6mm['distance (mm)'], 1)
rate6mm2 = np.polyfit(distanceTime6mm['Experiment 2'], distanceTime6mm['distance (mm)'], 1)
rate6mm3 = np.polyfit(distanceTime6mm['Experiment 4'], distanceTime6mm['distance (mm)'], 1)
rate6mm4 = np.polyfit(distanceTime6mm['Experiment 5'], distanceTime6mm['distance (mm)'], 1)

#plots of thermocouple distances to peak times
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(distanceTime6mm['Experiment 1'], distanceTime6mm['distance (mm)'], 'o')
axs[0,0].plot(dist, np.polyval(rate6mm1, dist))
axs[0,0].set_title('Experiment 1')
axs[0,1].plot(distanceTime6mm['Experiment 2'], distanceTime6mm['distance (mm)'], 'o')
axs[0,1].plot(dist, np.polyval(rate6mm2, dist))
axs[0,1].set_title('Experiment 2')
axs[1,0].plot(distanceTime6mm['Experiment 4'], distanceTime6mm['distance (mm)'], 'o')
axs[1,0].plot(dist, np.polyval(rate6mm3, dist))
axs[1,0].set_title('Experiment 3')
axs[1,1].plot(distanceTime6mm['Experiment 5'], distanceTime6mm['distance (mm)'], 'o')
axs[1,1].plot(dist, np.polyval(rate6mm4, dist))
axs[1,1].set_title('Experiment 4')


for ax in axs.flat:
    ax.set(xlabel='Distance (mm)', ylabel='Time (seconds)')
for ax in axs.flat:
    ax.label_outer()
plt.show()
'''
rates = []
rates.append(rate6mm1[0])
rates.append(rate6mm2[0])
rates.append(rate6mm3[0])
rates.append(rate6mm4[0])
print(rates)

#2mm test plot
plt.plot(test2mm1['Time (seconds)'], convertTotemp(test2mm1['Channel1']), label = 'Thermo -1' )
plt.plot(test2mm1['Time (seconds)'], convertTotemp(test2mm1['Channel0']),label = 'Thermo -2' )
plt.plot(test2mm1['Time (seconds)'], convertTotemp(test2mm1['Channel2']), label = 'Thermo -3' )
plt.plot(test2mm1['Time (seconds)'], convertTotemp(test2mm1['Channel3']), label = 'Thermo -4' )
plt.plot(test2mm1['Time (seconds)'], convertTotemp(test2mm1['Channel4']), label = 'Thermo -5' )
plt.plot(test2mm1['Time (seconds)'], convertTotemp(test2mm1['Channel5']), label = 'Thermo -6' )
plt.plot(test2mm1['Time (seconds)'], convertTotemp(test2mm1['Channel6']), label = 'Thermo -7' )
plt.plot(test2mm1['Time (seconds)'], convertTotemp(test2mm1['Channel7']), label = 'Thermo -8' )
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (Celcius)')
plt.title('Experiment 1')
plt.legend()
plt.show()


#6mm test plots

sixmm1 = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel0',color='red', label = 'Thermo - 8', ax = sixmm1)
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel1',color='green', label = 'Thermo - 7', ax =sixmm1)
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue', label = 'Thermo - 6', ax =sixmm1)
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel3',color='indigo', label = 'Thermo - 5', ax =sixmm1)
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel4',color='violet',label = 'Thermo - 4', ax =sixmm1)
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel5',color='black',label = 'Thermo - 3', ax =sixmm1)
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel6',color='orange', label = 'Thermo - 2',ax =sixmm1)
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel7',color='brown', label = 'Thermo - 1', ax =sixmm1)
plt.ylabel('Temperature (Celsius)')
plt.title('Experiment 1 (6mm)')
plt.show()

sixmm2 = plt.gca()
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel0',color='red',label = 'Thermo - 8', ax = sixmm2)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel1',color='green', label = 'Thermo - 7', ax =sixmm2)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue', label = 'Thermo - 6',ax =sixmm2)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel3',color='indigo', label = 'Thermo - 5',ax =sixmm2)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel4',color='violet',label = 'Thermo - 4', ax =sixmm2)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel5',color='black', label = 'Thermo - 3',ax =sixmm2)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel6',color='orange', label = 'Thermo - 2',ax =sixmm2)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel7',color='brown', label = 'Thermo - 1',ax =sixmm2)
plt.ylabel('Temperature (Celsius)')
plt.title('Experiment 2 (6mm)')
plt.show()
'''
'''
sixmm3 = plt.gca()
test6mm3.plot(kind='line',x='Time (seconds)',y='Channel0',color='red', ax = sixmm3)
test6mm3.plot(kind='line',x='Time (seconds)',y='Channel1',color='green', ax =sixmm3)
test6mm3.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue', ax =sixmm3)
test6mm3.plot(kind='line',x='Time (seconds)',y='Channel3',color='indigo', ax =sixmm3)
test6mm3.plot(kind='line',x='Time (seconds)',y='Channel4',color='violet', ax =sixmm3)
test6mm3.plot(kind='line',x='Time (seconds)',y='Channel5',color='black', ax =sixmm3)
test6mm3.plot(kind='line',x='Time (seconds)',y='Channel6',color='orange', ax =sixmm3)
test6mm3.plot(kind='line',x='Time (seconds)',y='Channel7',color='brown', ax =sixmm3)
plt.ylabel('Temperature')
plt.title('Experiment fail (6mm)')
plt.show()

sixmm4 = plt.gca()
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel0',color='red', label = 'Thermo - 8', ax = sixmm4)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel1',color='green', label = 'Thermo - 7', ax =sixmm4)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue', label = 'Thermo - 6',ax =sixmm4)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel3',color='indigo', label = 'Thermo - 5',ax =sixmm4)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel4',color='violet', label = 'Thermo - 4',ax =sixmm4)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel5',color='black', label = 'Thermo - 3',ax =sixmm4)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel6',color='orange', label = 'Thermo - 2',ax =sixmm4)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel7',color='brown',label = 'Thermo - 1', ax =sixmm4)
plt.ylabel('Temperature (Celsius)')
plt.title('Experiment 3 (6mm)')
plt.show()

sixmm5 = plt.gca()
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel0',color='red', label = 'Thermo - 8', ax = sixmm5)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel1',color='green',label = 'Thermo - 7', ax =sixmm5)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue', label = 'Thermo - 6',ax =sixmm5)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel3',color='indigo', label = 'Thermo - 5',ax =sixmm5)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel4',color='violet', label = 'Thermo - 4',ax =sixmm5)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel5',color='black', label = 'Thermo - 3',ax =sixmm5)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel6',color='orange', label = 'Thermo - 2',ax =sixmm5)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel7',color='brown', label = 'Thermo - 1',ax =sixmm5)
plt.ylabel('Temperature (Celsius)')
plt.title('Experiment 4 (6mm)')
plt.show()


#tenmm test plots

tenmm1 = plt.gca()
test10mm1.plot(kind='line',x='Time (seconds)',y='Channel0',color='red',label = 'Thermo -8', ax = tenmm1)
test10mm1.plot(kind='line',x='Time (seconds)',y='Channel1',color='green',label = 'Thermo -7', ax =tenmm1)
test10mm1.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue',label = 'Thermo -6', ax =tenmm1)
test10mm1.plot(kind='line',x='Time (seconds)',y='Channel3',color='indigo', label = 'Thermo -5',ax =tenmm1)
test10mm1.plot(kind='line',x='Time (seconds)',y='Channel4',color='violet', label = 'Thermo -4',ax =tenmm1)
test10mm1.plot(kind='line',x='Time (seconds)',y='Channel5',color='black', label = 'Thermo -3',ax =tenmm1)
test10mm1.plot(kind='line',x='Time (seconds)',y='Channel6',color='orange', label = 'Thermo -2',ax =tenmm1)
test10mm1.plot(kind='line',x='Time (seconds)',y='Channel7',color='brown', label = 'Thermo -1',ax =tenmm1)
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (Celsius)')
plt.title('Experiment 1 (10mm)')
plt.show()

tenmm2 = plt.gca()
test10mm2.plot(kind='line',x='Time (seconds)',y='Channel0',color='red',label = 'Thermo -8', ax = tenmm2)
test10mm2.plot(kind='line',x='Time (seconds)',y='Channel1',color='green',label = 'Thermo -7', ax =tenmm2)
test10mm2.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue',label = 'Thermo -6', ax =tenmm2)
test10mm2.plot(kind='line',x='Time (seconds)',y='Channel3',color='indigo', label = 'Thermo -5', ax =tenmm2)
test10mm2.plot(kind='line',x='Time (seconds)',y='Channel4',color='violet', label = 'Thermo -4', ax =tenmm2)
test10mm2.plot(kind='line',x='Time (seconds)',y='Channel5',color='black', label = 'Thermo -3', ax =tenmm2)
test10mm2.plot(kind='line',x='Time (seconds)',y='Channel6',color='orange', label = 'Thermo -2', ax =tenmm2)
test10mm2.plot(kind='line',x='Time (seconds)',y='Channel7',color='brown', label = 'Thermo -1', ax =tenmm2)
plt.xlabel('Time (seconds)')
plt.ylabel('Temperature (Celsius)')
plt.title('Experiment 2 (10mm)')
plt.show()


#6mm thermocouple comparison


channel06mm = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel0',color='red', label='test 1', ax = channel06mm)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel0',color='green',label='test 2', ax = channel06mm)
#test6mm3.plot(kind='line',x='Time (seconds)',y='Channel0',color='blue', label='test 1',ax = channel06mm)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel0',color='blue', label='test 3',ax = channel06mm)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel0',color='black', label='test 4',ax = channel06mm)
plt.ylabel('Temperature (Celsius)')
plt.title('Thermocouple 8 - Channel 0')
plt.show()

channel16mm = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel1',color='red',label='test 1', ax = channel16mm)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel1',color='green',label='test 2', ax = channel16mm)
#test6mm3.plot(kind='line',x='Time (seconds)',y='Channel1',color='blue', ax = channel16mm)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel1',color='blue', label='test 3', ax = channel16mm)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel1',color='black', label='test 4',ax = channel16mm)
plt.ylabel('Temperature (Celsius)')
plt.title('Thermocouple 7 - Channel 1')
plt.show()

channel26mm = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel2',color='red',label='test 1', ax = channel26mm)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel2',color='green',label='test 2', ax = channel26mm)
#test6mm3.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue', ax = channel26mm)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel2',color='blue', label='test 3', ax = channel26mm)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel2',color='black', label='test 4',ax = channel26mm)
plt.ylabel('Temperature (Celsius)')
plt.title('Thermocouple 6 - Channel 2')
plt.show()

channel36mm = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel3',color='red',label='test 1', ax = channel36mm)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel3',color='green',label='test 2', ax = channel36mm)
#test6mm3.plot(kind='line',x='Time (seconds)',y='Channel3',color='blue', ax = channel36mm)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel3',color='blue', label='test 3', ax = channel36mm)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel3',color='black',label='test 4', ax = channel36mm)
plt.ylabel('Temperature (Celsius)')
plt.title('Thermocouple 5 - Channel 3')
plt.show()

channel46mm = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel4',color='red',label='test 1', ax = channel46mm)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel4',color='green', label='test 2', ax = channel46mm)
#test6mm3.plot(kind='line',x='Time (seconds)',y='Channel4',color='blue', ax = channel46mm)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel4',color='blue', label='test 3', ax = channel46mm)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel4',color='black',label='test 4', ax = channel46mm)
plt.ylabel('Temperature (Celsius)')
plt.title('Thermocouple 4 - Channel 4')
plt.show()

channel56mm = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel5',color='red',label='test 1', ax = channel56mm)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel5',color='green', label='test 2', ax = channel56mm)
#test6mm3.plot(kind='line',x='Time (seconds)',y='Channel5',color='blue', ax = channel56mm)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel5',color='blue', label='test 3', ax = channel56mm)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel5',color='black', label='test 4',ax = channel56mm)
plt.ylabel('Temperature (Celsius)')
plt.title('Thermocouple 3 - Channel 5')
plt.show()

channel66mm = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel6',color='red', label='test 1',ax = channel66mm)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel6',color='green',label='test 2', ax = channel66mm)
#test6mm3.plot(kind='line',x='Time (seconds)',y='Channel6',color='blue', ax = channel66mm)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel6',color='blue', label='test 3', ax = channel66mm)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel6',color='black', label='test 4', ax = channel66mm)
plt.ylabel('Temperature (Celsius)')
plt.title('Thermocouple 2 - Channel 6')
plt.show()

channel76mm = plt.gca()
test6mm1.plot(kind='line',x='Time (seconds)',y='Channel7',color='red',label='test 1', ax = channel76mm)
test6mm2.plot(kind='line',x='Time (seconds)',y='Channel7',color='green',label='test 2', ax = channel76mm)
#test6mm3.plot(kind='line',x='Time (seconds)',y='Channel7',color='blue', ax = channel76mm)
test6mm4.plot(kind='line',x='Time (seconds)',y='Channel7',color='blue',label = 'test 3', ax = channel76mm)
test6mm5.plot(kind='line',x='Time (seconds)',y='Channel7',color='black', label = 'test 4',ax = channel76mm)
plt.ylabel('Temperature (Celsius)')
plt.title('Thermocouple 1 - Channel 7')
plt.show()
'''