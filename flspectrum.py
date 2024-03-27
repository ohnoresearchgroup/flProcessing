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
        
    def bgSubtractAndSmooth(self,smoothparam):
        self.datacorr = self.data['counts'] - self.bgdata['counts']
        self.datacorrsmooth = savgol_filter(self.datacorr, smoothparam, 3)
        
        self.smoothmax = self.data['wl'][self.datacorrsmooth.argmax()]
        self.smoothmaxuncalib = self.data['wluncorr'][self.datacorrsmooth.argmax()]

        plt.figure()
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Fluorescence Intensity [counts]")
        plt.title('Smoothed')
        plt.plot(self.data['wl'], self.datacorr)
        plt.plot(self.data['wl'], self.datacorrsmooth)
        
        def findIndex(array, target_value):
            minarray = abs(array-target_value)
            return np.argmin(minarray)
        
        centroid = np.sum(self.datacorrsmooth*self.data['wl'])/np.sum(self.datacorrsmooth)
        indexCentroid = findIndex(self.data['wl'],centroid)
        self.centroid = centroid
        self.indexCentroid = indexCentroid
        
        self.datacorrsmoothnorm = self.datacorrsmooth/np.max(self.datacorrsmooth)

        maxidx = findIndex(self.datacorrsmoothnorm, 1)
        leftHalfIndex = findIndex(self.datacorrsmoothnorm[0:maxidx],0.5)
        rightHalfIndex = findIndex(self.datacorrsmoothnorm[maxidx:],0.5)+maxidx
        self.width = (self.data['wl'][rightHalfIndex]-self.data['wl'][leftHalfIndex])
        print('FWHM= ',self.width)


        plt.figure()
        plt.plot(self.data['wl'],self.datacorrsmoothnorm)
        plt.plot([self.data['wl'][leftHalfIndex],self.data['wl'][leftHalfIndex]],[0,1])
        plt.plot([self.data['wl'][rightHalfIndex],self.data['wl'][rightHalfIndex]],[0,1])
        plt.plot([self.data['wl'][maxidx],self.data['wl'][maxidx]],[0,1])
        plt.plot([self.data['wl'][indexCentroid],self.data['wl'][indexCentroid ]],[0,1])

    def fitGuassian(self,p0):
        def gaussian(x,x0,a,h):
            return h*np.exp(-a*(x-x0)*(x-x0))

        params = curve_fit(gaussian, self.data['wl'], self.datacorr, p0=p0) 
        paramsuncalib = curve_fit(gaussian, self.data['wluncorr'], self.datacorr, p0=p0) 

        self.gausscenter = params[0][0]
        self.gausscenteruncalib = paramsuncalib[0][0]

        plt.figure()
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Fluorescence Intensity [counts]")
        plt.title("Gaussian Fit")
        plt.plot(self.data['wl'], self.datacorr)
        fit = gaussian(self.data['wl'],params[0][0],params[0][1],params[0][2])
        plt.plot(self.data['wl'], fit)
    