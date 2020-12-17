import cv2
import pydicom
import numpy as np
import os
from glob import glob
import json
import pickle
import pickle5
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
            self.database = pickle5.load(pklfile)
            
            

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
    
    
    
    def returnGuidewire(self, p, set, drawCat=False, tipOnly=True, thresh=4):
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
            first = True
            for xy in tData:
                xstr = xy.split(' ')[0]
                ystr = xy.split(' ')[1]
                x = int(float(xstr))
                y = int(float(ystr))
                if x > 511: x = 511
                if y > 511: y = 511
                if first:
                    first = False
                    dist = np.linalg.norm(np.array(temp[-1]) - np.array([x, y]))
                    if dist > thresh:
                        temp.reverse()
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
    
    
def returnReverse(first, last, vgwMean):
    if type(first) == list:
        first = np.array(first)
    if type(last) == list:
        last = np.array(last)
        
    dist_f = np.linalg.norm(first - vgwMean)
    dist_l = np.linalg.norm(last - vgwMean)
    
    if dist_f > dist_l:
        return True
    else:
        return False
    
    
def returnGuidewireNew(path, vgwMean, drawCat=False, tipOnly=True):
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

        tempB = []
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
                    tempB.append([x, y])
                bodF.close()

                if returnReverse(tempB[0], tempB[-1], vgwMean):
                    tempB.reverse()

        tipF = open(tpath, 'r')
        tData = tipF.readlines()
        tempT = []
        for xy in tData:
            xstr = xy.split(' ')[0]
            ystr = xy.split(' ')[1]
            x = int(float(xstr))
            y = int(float(ystr))
            if x > 511: x = 511
            if y > 511: y = 511
            tempT.append([x, y])
        tipF.close()

        if returnReverse(tempT[0], tempT[-1], vgwMean):
            tempT.reverse()

        tempB.extend(tempT)

        gws.append(tempB)

    return gws


def makeAngioOverlay(root):
    angio_dirs = glob(os.path.join(root, 'Angio/*.png'))
    #print(angio_dirs)
    #imgs = []
    added = np.zeros((512,512), dtype=np.float32)
    for i in range(len(angio_dirs)):
        temp = cv2.imread(angio_dirs[i], cv2.IMREAD_GRAYSCALE)
        #imgs.append(cv2.imread(angio_dirs[i], cv2.IMREAD_GRAY))
        added += temp
        
    norm = (added / len(angio_dirs)).astype(np.uint8)
    
    return norm


def drawWireInit(angio, wires, color=(0,0,255), L=1, circle=0, th=-1, isFirst=False, trans=None):
    if angio.shape[-1] != 3:
        angio = cv2.cvtColor(angio, cv2.COLOR_GRAY2BGR)
    
    if not isFirst:
        for i in range(len(wires)):
            wire = wires[i]
            #wire.sort()
            for j in range(L):
                if trans is None:
                    x = wire[j][0]
                    y = wire[j][1]
                else:
                    x = wire[j][0] - trans[0]
                    y = wire[j][1] - trans[1]

                if L == 1 and circle:
                    angio = cv2.circle(angio, (x,y), circle, color, th)
                else:
                    angio[y,x] = color
                    
    else:
        for i in range(len(wires)):
            if trans is None:
                x = wires[i][0]
                y = wires[i][1]
            else:
                x = wires[i][0] - trans[0]
                y = wires[i][1] - trans[1]
            
            if circle == 0:
                circle = 2
            
            angio = cv2.circle(angio, (x,y), circle, color, th)
            
    return angio


def loadFirstPoint(p, s, db_path='../Medical_final_labels/setting.pkl', save='Translated', returnAll=False):
    db = DataLoading(db_path, save)
    
    vgws = []
    for key in db.database[p][s].keys():
        if type(key) == int:
            line = db.database[p][s][key]['SelectedLabels'][0][1]
            vgws.append(line)
    
    vgw_first = []
    for vgw in vgws:
        vgw_first.append(vgw[0])
    vgw_first = np.array(vgw_first)
    vgw_mean = np.mean(vgw_first, axis=0)
    
    for i in range(len(vgws)):
        vgw = vgws[i]
        first = np.array(vgw[0])
        last = np.array(vgw[-1])
        
        d_first = np.linalg.norm(first - vgw_mean)
        d_last = np.linalg.norm(last - vgw_mean)
        
        if d_first > d_last:
            vgws[i].reverse()
    
    vgw_first = []
    for vgw in vgws:
        vgw_first.append(vgw[0])
    vgw_first = np.array(vgw_first)
    vgw_mean_new = np.mean(vgw_first, axis=0)
    
    if returnAll:
        return vgws, vgw_mean_new
    else:
        return vgw_first, vgw_mean_new