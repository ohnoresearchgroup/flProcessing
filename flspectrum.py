import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks, argrelmax
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import copy

class FlSpectrum():
    def __init__(self,path, name, bgname,RH):
        self.path = path
        self.name = name
        self.bgname = bgname
        self.RH = RH
        self.data = pd.read_csv(path + name, skiprows = 43, delimiter = "\t", names = ['wl', 'counts','a']).drop('a',axis=1)
        self.bgdata = pd.read_csv(path + bgname, skiprows = 43, delimiter = "\t", names = ['wl', 'counts','a']).drop('a',axis=1)
        self.calibrated = False
        self.gaussianfitted = False
        
        
    def printData(self):
        print(self.data)
        
    def printBGdata(self):
        print(self.bgdata)
        
    def calibrate(self,prominence,dataoffset,bgoffset):
        peaks, _ = find_peaks(self.bgdata['counts'], prominence = prominence)
        self.bgdata['wluncorr'] = copy.deepcopy(self.bgdata['wl'])
        print('bg shift ',(self.bgdata['wl'][peaks[0]]-532+bgoffset))
        self.bgdata['wl'] = self.bgdata['wl']-(self.bgdata['wl'][peaks[0]]-532+bgoffset)

        self.data['wluncorr'] = copy.deepcopy(self.data['wl'])
        peaks, _ = find_peaks(self.data['counts'], prominence = prominence)
        print('data shift ', (self.data['wl'][peaks[0]]-532+dataoffset))
        self.data['wl'] = self.data['wl']-(self.data['wl'][peaks[0]]-532+dataoffset)

        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Fluorescence Intensity [counts]")
        plt.plot(self.bgdata['wl'], self.bgdata['counts'])
        plt.plot(self.data['wl'], self.data['counts']) 
        plt.plot([532,532],[0,2e4])
        self.calibrated = True
        
    def calibrateSquare(self,dataLims=[510,560]):
        self.bgdata['wluncorr'] = copy.deepcopy(self.bgdata['wl'])
        self.data['wluncorr'] = copy.deepcopy(self.data['wl'])
        
        x = self.bgdata['wluncorr']
        y = self.bgdata['counts']
        d = np.gradient(y,x)
        idx1 = np.abs(x - dataLims[0]).argmin()
        idx2 = np.abs(x - dataLims[1]).argmin()
        left = d[idx1:idx2].argmax()
        right = d[idx1:idx2].argmin()
        shift = np.mean([x[left+idx1],x[right+idx1]])
        
        print('bg shift ',(shift-532))
        self.bgdata['wl'] = self.bgdata['wl']-(shift-532)
        self.data['wl'] = self.data['wl']-(shift-532)

        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Fluorescence Intensity [counts]")
        plt.plot(self.bgdata['wl'], self.bgdata['counts'])
        plt.plot(self.data['wl'], self.data['counts']) 
        plt.plot([532,532],[0,2e4])
        self.calibrated = True
        
    def bgSubtractAndSmooth(self,smoothparam):
        self.datasub = self.data['counts'] - self.bgdata['counts']
        self.datasubsmooth = savgol_filter(self.datasub, smoothparam, 3)
        
        self.smoothmax = self.data['wl'][self.datasubsmooth.argmax()]
        if self.calibrated == True:
            self.smoothmaxuncalib = self.data['wluncorr'][self.datasubsmooth.argmax()]
       
        plt.figure()
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Fluorescence Intensity [counts]")
        plt.title('Smoothed')
        plt.plot(self.data['wl'], self.bgdata['counts'])
        plt.plot(self.data['wl'], self.data['counts'])
        
        plt.figure()
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Fluorescence Intensity [counts]")
        plt.title('Smoothed')
        plt.plot(self.data['wl'], self.datasub)
        plt.plot(self.data['wl'], self.datasubsmooth)
        
        def findIndex(array, target_value):
            minarray = abs(array-target_value)
            return np.argmin(minarray)
        
        centroid = np.sum(self.datasubsmooth*self.data['wl'])/np.sum(self.datasubsmooth)
        indexCentroid = findIndex(self.data['wl'],centroid)
        self.centroid = centroid
        self.indexCentroid = indexCentroid
        
        self.datasubsmoothnorm = self.datasubsmooth/np.max(self.datasubsmooth)

        maxidx = findIndex(self.datasubsmoothnorm, 1)
        leftHalfIndex = findIndex(self.datasubsmoothnorm[0:maxidx],0.5)
        rightHalfIndex = findIndex(self.datasubsmoothnorm[maxidx:],0.5)+maxidx
        self.width = (self.data['wl'][rightHalfIndex]-self.data['wl'][leftHalfIndex])
        print('FWHM= ',self.width)


        plt.figure()
        plt.plot(self.data['wl'],self.datasubsmoothnorm)
        plt.plot([self.data['wl'][leftHalfIndex],self.data['wl'][leftHalfIndex]],[0,1])
        plt.plot([self.data['wl'][rightHalfIndex],self.data['wl'][rightHalfIndex]],[0,1])
        plt.plot([self.data['wl'][maxidx],self.data['wl'][maxidx]],[0,1])
        plt.plot([self.data['wl'][indexCentroid],self.data['wl'][indexCentroid ]],[0,1])

    def fitGuassian(self,p0):
        def gaussian(x,x0,a,h):
            return h*np.exp(-a*(x-x0)*(x-x0))

        params = curve_fit(gaussian, self.data['wl'], self.datasub, p0=p0)
        self.gausscenter = params[0][0]
        
        if self.calibrated == True:
            paramsuncalib = curve_fit(gaussian, self.data['wluncorr'], self.datasub, p0=p0) 
            self.gausscenteruncalib = paramsuncalib[0][0]

        plt.figure()
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Fluorescence Intensity [counts]")
        plt.title("Gaussian Fit")
        plt.plot(self.data['wl'], self.datasub)
        fit = gaussian(self.data['wl'],params[0][0],params[0][1],params[0][2])
        plt.plot(self.data['wl'], fit)
        self.gaussianfitted = True
        
    def getSmoothMax(self):
        return self.smoothmax
    
    def getSmoothMaxUncalib(self):
        if self.calibrated == True:
            return self.smoothmaxuncalib
        else:
            print('not calibrated so no uncalibrated Gauss center')
            return
    
    def getGaussCenter(self):
        if self.gaussianfitted == True:
            return self.gausscenter
        else:
            print('not gaussian fitted')
            return
    
    def getGaussCenterUncalib(self):
        if (self.calibrated == True) and (self.gaussianfitted == True):
            return self.gausscenteruncalib
        else:
            print('not calibrated or not gauss fitted so no uncalibrated Gauss center')
            return
    