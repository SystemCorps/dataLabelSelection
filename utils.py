import cv2
import pydicom
import numpy as np
import os
from glob import glob
import json
import pickle
#import pickle5
import platform
import gzip
import sys
import random

class DataLoading():
    def __init__(self, dbPath, genSavePath, angioRoot='.'+os.sep+'angio_labels', wiringRoot='.'+os.sep+'gw_labels', dcmRoot='.'+os.sep+'dicoms', imgSize=(512,512, 3)):

        self.dbPath = dbPath
        self.genSavePath = genSavePath

        self.angioRoot = angioRoot
        self.wiringRoot = wiringRoot
        self.dcmRoot = dcmRoot
        self.imgSize = imgSize

        self.steps = 50
        self.minLen = 30

        self.TotalList, self.totalCount = self.getIntersection()

        with open(self.dbPath, 'rb') as pklfile:
            self.database = pickle.load(pklfile)
            
            

    def savePathName(self, patient, set, folder, frame, subIdx=None, stepIdx=None, angio=False):
        temp = os.path.join(self.genSavePath, "{0:02d}".format(patient))
        temp = os.path.join(temp, set)
        temp = os.path.join(temp, folder)
        if subIdx is not None:
            temp = os.path.join(temp, "{0:02d}".format(subIdx))
        head = temp
        if angio:
            name = "{0:05d}.png".format(frame)
        elif stepIdx is not None:
            name = "{0:03d}_".format(frame) + "{0:05d}.png".format(stepIdx)
        else:
            name = "{0:05d}.png".format(frame)
        full = os.path.join(head, name)
        return head, name, full
    
    
    def genSplitted(self, line):
        totalLen = len(line)
        numPerStep = int(np.ceil((totalLen - self.minLen) / (self.steps - 1)))

        lines = []

        for i in range(self.steps):
            if i == 0:
                endIdx = self.minLen
            else:
                endIdx = self.minLen + i * numPerStep

            if endIdx >= totalLen - 1:
                temp = line
            else:
                temp = line[:endIdx]
            lines.append(temp)

        return lines
    
    
    def makeSingleLine(self, patient, set, frame, idx=0):

        lineSet = self.database[patient][set][frame]['SelectedLabels'][idx]

        for key in lineSet.keys():
            if key == 1:
                output = lineSet[key]
            elif key==2:
                start2nd = np.array(lineSet[key][0])
                line1st = np.array(output)
                norm = np.linalg.norm(line1st - start2nd, axis=1)
                line1stEndIdx = np.argmin(norm)

                output = output[:line1stEndIdx]
                output.extend(lineSet[key])
        
        return output
    
    
    
    def drawSavedLabel(self, label, base=None, size=(512,512), xyswap=False, value=[255, 0, 0]):
        if base is None:
            if type(value) == int:
                base = np.ones(size, dtype=np.uint8) * 255
            elif type(value) == list:
                newSize = list(size)
                newSize.append(len(value))
                base = np.ones(tuple(newSize), dtype=np.uint8) * 255

        for x, y in label:
            if xyswap:
                base[y, x] = value
            else:
                base[x, y] = value

        return base
    
    
    def genProgress(self, line, tipL, tipStd, size=(512,512), xyswap=True, value=0):
        interval = (len(line) - tipL) // self.steps
        imgs = []
        for i in range(self.steps):
            start = int(i * interval)
            end = start + int(np.random.normal(tipL, tipStd))
            if end > (len(line) - 1):
                end = -1
                
            base = np.ones(size, dtype=np.uint8) * 255

            xy = line[start:end]
            for x, y in xy:
                if xyswap:
                    base[y,x] = value
                else:
                    base[x,y] = value
            
            imgs.append(base)
            
        return imgs
    
    
    
    def drawWire(self, wire, base=None, size=(512,512), xyswap=False, value=[0, 255, 0]):
        if base is None:
            base = np.ones(size, dtype=np.uint8) * 255

        for x, y in wire:
            if xyswap:
                base[y, x] = value
            else:
                base[x, y] = value

        return base
    
    
    def setDirectory(self, p, set, root='.'+os.sep+'angio_labels'):
        patient = "{0:02d}".format(p)
        path = os.path.join(root, patient)
        path = os.path.join(path, set)
        return path
    
    
    
    def returnGuidewire(self, p, set, drawCat=False, tipOnly=True):
        path = self.setDirectory(p, set, root='.'+os.sep+'gw_labels')
        tips = glob(os.path.join(path, '*tip.txt'))
        bodies = glob(os.path.join(path, '*body.txt'))
        cats = glob(os.path.join(path, '*ter.txt'))

        tips.sort()
        bodies.sort()
        cats.sort()

        gws = []
        last = tips[-1]
        lastName = os.path.split(last)[1]
        lastFrame = int(lastName.split('_')[0])
        blist = [None] * (lastFrame + 1)
        for bpath in bodies:
            name = os.path.split(bpath)[1]
            frame = int(name.split('_')[0])
            blist[frame] = bpath

        for tpath in tips:
            name = os.path.split(tpath)[1]
            frame = name.split('_')[0]
            nframe = int(frame)

            temp = []
            if not tipOnly:
                if blist[nframe]:
                    bodF = open(blist[nframe], 'r')
                    bData = bodF.readlines()
                    for xy in bData:
                        xstr = xy.split(' ')[0]
                        ystr = xy.split(' ')[1]
                        x = int(float(xstr))
                        y = int(float(ystr))
                        if x > 511: x = 511
                        if y > 511: y = 511
                        temp.append([x, y])
                    bodF.close()

            tipF = open(tpath, 'r')
            tData = tipF.readlines()
            for xy in tData:
                xstr = xy.split(' ')[0]
                ystr = xy.split(' ')[1]
                x = int(float(xstr))
                y = int(float(ystr))
                if x > 511: x = 511
                if y > 511: y = 511
                temp.append([x, y])
            tipF.close()

            gws.append(temp)

        return gws
    
    
    def drawGuidewires(self, patient, set, size=(512,512), xyswap=True, value=0):
        guidewires = self.returnGuidewire(patient, set)
        outputs = []

        for gw in guidewires:
            base = np.ones(size, dtype=np.uint8) * 255
            for x, y in gw:
                if xyswap:
                    base[y, x] = value
                else:
                    base[x, y] = value
            outputs.append(base)

        return outputs
    
    
    def patientSet(self, root='.'+os.sep+'angio_labels' , file='**'+os.sep+'*_level1.txt'):
        dirs = glob(os.path.join(root, file), recursive=True)
        dirs.sort()

        plists = {}

        for subpath in dirs:
            splitted = subpath.split(os.sep)
            patient = int(splitted[2])
            if not (patient in plists.keys()):
                plists[patient] = []

            if len(splitted) == 5:
                patient = int(splitted[2])
                set = splitted[3]
                if not (set in plists[patient]):
                    plists[patient].append(set)

        return plists
    
    
    
    def getIntersection(self):
        angio = self.patientSet(self.angioRoot)
        wiring = self.patientSet(self.wiringRoot, '**'+os.sep+'*.txt')

        intersection = {}

        angioKeys = list(angio.keys())
        wiringKeys = list(wiring.keys())

        interKeys = list(set(angioKeys) & set(wiringKeys))
        count = 0
        for key in interKeys:
            angioSetList = angio[key]
            wiringSetList = wiring[key]
            interSetList = list(set(angioSetList) & set(wiringSetList))
            
            if interSetList:
                intersection[key] = interSetList
                count+=1
                
        return intersection, count