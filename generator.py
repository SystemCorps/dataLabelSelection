import cv2
import pydicom
import numpy as np
import os
import sys
from glob import glob
import json
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt

class ResultWindow(QDialog):
    def __init__(self, parent):
        super(ResultWindow, self).__init__(parent)
        uic.loadUi('resultwindow.ui', self)
        self.show()

class LabelSelection(QMainWindow):

    def __init__(self, uiPath='mainwindow.ui', angioRoot='./angio_labels', wiringRoot='./gw_labels', dcmRoot='./dicoms', imgSize=(512,512, 3)):
        super(LabelSelection, self).__init__()
        uic.loadUi(uiPath, self)
        self.show()
        self.angioRoot = angioRoot
        self.wiringRoot = wiringRoot
        self.dcmRoot = dcmRoot
        self.imgSize = imgSize

        self.TotalList = self.getIntersection()

        # Selectable
        self.showSel = [1, 0, 0, 1]
        self.levelShowSel = {1:0, 2:0, 3:0, 4:0}
        self.levelSel = {1:0, 2:0, 3:0, 4:0}
        self.levelColor = {1:[255, 0, 0], 2:[255, 204, 51], 3:[51, 255, 255], 4:[0, 0, 255]}

        self.currentPatientIdx = 0
        self.currentPatient = list(self.TotalList.keys())[self.currentPatientIdx]
        self.currentSetList = self.TotalList[self.currentPatient]
        self.currentSetIdx = 0
        self.currentSet = self.currentSetList[self.currentSetIdx]
        self.currentFrameList = self.frameList(self.currentPatient, self.currentSet)
        self.currentFrameIdx = 0
        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.frameOffset = 0

        self.currentLabel = self.returnLabels(self.currentPatient, self.currentSet, self.currentFrame)

        self.wiringPix = self.returnGuidewire(self.currentPatient, self.currentSet)
        self.currentWiringFrame = 0

        self.currentDCMImgs, self.currentDCMPaths = self.getDicoms(self.currentPatient,
                                                                   self.currentFrame + self.frameOffset)
        self.currentDCMIdx = 0

        self.database = dict.fromkeys(list(self.TotalList.keys()))

        self.sliderUpdate()

        self.pnumDisp.setText("{}".format(self.currentPatient))
        self.psetDisp.setText("{}".format(self.currentSet))

        # Buttons
        self.dcmNumNext.clicked.connect(self.setDCMNumNext)
        self.dcmNumPrev.clicked.connect(self.setDCMNumPrev)
        #self.genData.clicked.connect(self.generateDataset)
        #self.genPath.clicked.connect(self.setGenSavePath)
        self.level1Next.clicked.connect(self.setL1Next)
        self.level1Prev.clicked.connect(self.setL1Prev)
        self.level2Next.clicked.connect(self.setL2Next)
        self.level2Prev.clicked.connect(self.setL2Prev)
        self.level3Next.clicked.connect(self.setL3Next)
        self.level3Prev.clicked.connect(self.setL3Prev)
        self.level4Next.clicked.connect(self.setL4Next)
        self.level4Prev.clicked.connect(self.setL4Prev)
        self.pnumNext.clicked.connect(self.setPNext)
        self.pnumPrev.clicked.connect(self.setPPrev)
        self.psetNext.clicked.connect(self.setPSNext)
        self.psetPrev.clicked.connect(self.setPSPrev)
        self.resetDICOM.clicked.connect(self.resetDCMset)
        #self.delVG.clicked.connect(self.resetVGset)
        #self.savePath.clicked.connect(self.setSavePath)
        self.setDICOM.clicked.connect(self.setDCMset)
        #self.setVG.clicked.connect(self.setVGset)
        self.wiringNext.clicked.connect(self.setWireNext)
        self.wiringPrev.clicked.connect(self.setWirePrev)
        self.angioFrameNext.clicked.connect(self.setAngioFrameNext)
        self.angioFramePrev.clicked.connect(self.setAngioFramePrev)
        self.dcmOffsetUp.clicked.connect(self.setDCMOffsetUp)
        self.dcmOffsetDown.clicked.connect(self.setDCMOffsetDown)
        self.showResult.clicked.connect(self.showResultWindow)

        # checkbox
        self.showDICOM.stateChanged.connect(self.showing)
        self.showVG.stateChanged.connect(self.showing)
        self.showWiring.stateChanged.connect(self.showing)
        self.showEvery.stateChanged.connect(self.showing)
        self.level1Check.stateChanged.connect(self.levelCheck)
        self.level2Check.stateChanged.connect(self.levelCheck)
        self.level3Check.stateChanged.connect(self.levelCheck)
        self.level4Check.stateChanged.connect(self.levelCheck)
        # slider
        self.wiringSlider.valueChanged[int].connect(self.sliderWire)

        # Display
        self.dcmInfoDisp()
        self.frameInfoDisp()
        self.levelInfoDisp()
        # Image Show
        self.updateImage()

    def showResultWindow(self):
        ResultWindow(self)


    #def makeResult(self):


    def setVGset(self):
        self.database[self.currentPatient][self.currentSet][self.currentFrame] = {}
        try:
            self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabelIdx'].append(self.levelShowSel)
            self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels'].append([self.currentLabel[1][self.levelShowSel[1]],
                                                                                                             self.currentLabel[2][self.levelShowSel[2]],
                                                                                                             self.currentLabel[3][self.levelShowSel[3]],
                                                                                                             self.currentLabel[4][self.levelShowSel[4]]])

        except:






    def setDCMset(self):
        if self.database[self.currentPatient]:
            self.database[self.currentPatient][self.currentSet] = {'DCMPath':self.currentDCMPaths[self.currentDCMIdx]}
        else:
            self.database[self.currentPatient] = {self.currentSet: {'DCMPath':self.currentDCMPaths[self.currentDCMIdx]}}

    def resetDCMset(self):
        try:
            del(self.database[self.currentPatient][self.currentSet])
        except (KeyError, TypeError):
            pass


    def setL1Next(self):
        idx = 1
        self.levelSel[idx] += 1
        if self.levelSel[idx] > len(self.currentLabel[idx]) - 1:
            self.levelSel[idx] = len(self.currentLabel[idx]) - 1
        self.levelInfoDisp()
        self.updateImage()

    def setL1Prev(self):
        idx = 1
        self.levelSel[idx] -= 1
        if self.levelSel[idx] < 0:
            self.levelSel[idx] = 0
        self.levelInfoDisp()
        self.updateImage()

    def setL2Next(self):
        idx = 2
        self.levelSel[idx] += 1
        if self.levelSel[idx] > len(self.currentLabel[idx]) - 1:
            self.levelSel[idx] = len(self.currentLabel[idx]) - 1
        self.levelInfoDisp()
        self.updateImage()

    def setL2Prev(self):
        idx = 2
        self.levelSel[idx] -= 1
        if self.levelSel[idx] < 0:
            self.levelSel[idx] = 0
        self.levelInfoDisp()
        self.updateImage()

    def setL3Next(self):
        idx = 3
        self.levelSel[idx] += 1
        if self.levelSel[idx] > len(self.currentLabel[idx]) - 1:
            self.levelSel[idx] = len(self.currentLabel[idx]) - 1
        self.levelInfoDisp()
        self.updateImage()

    def setL3Prev(self):
        idx = 3
        self.levelSel[idx] -= 1
        if self.levelSel[idx] < 0:
            self.levelSel[idx] = 0
        self.levelInfoDisp()
        self.updateImage()

    def setL4Next(self):
        idx = 4
        self.levelSel[idx] += 1
        if self.levelSel[idx] > len(self.currentLabel[idx]) - 1:
            self.levelSel[idx] = len(self.currentLabel[idx]) - 1
        self.levelInfoDisp()
        self.updateImage()

    def setL4Prev(self):
        idx = 4
        self.levelSel[idx] -= 1
        if self.levelSel[idx] < 0:
            self.levelSel[idx] = 0
        self.levelInfoDisp()
        self.updateImage()


    def levelInfoDisp(self):
        self.level1NumDisp.setText("{}".format(self.levelSel[1]))
        self.level2NumDisp.setText("{}".format(self.levelSel[2]))
        self.level3NumDisp.setText("{}".format(self.levelSel[3]))
        self.level4NumDisp.setText("{}".format(self.levelSel[4]))

    def levelCheck(self):
        if self.level1Check.isChecked(): self.levelShowSel[1] = 1
        else: self.levelShowSel[1] = 0
        if self.level2Check.isChecked(): self.levelShowSel[2] = 1
        else: self.levelShowSel[2] = 0
        if self.level3Check.isChecked(): self.levelShowSel[3] = 1
        else: self.levelShowSel[3] = 0
        if self.level4Check.isChecked(): self.levelShowSel[4] = 1
        else: self.levelShowSel[4] = 0

        self.updateImage()

    def setLevelButtons(self):
        label = self.currentLabel
        if self.showEvery.isChecked():
            bool = False
        else:
            bool = True

        if label[1]:
            self.level1Next.setEnabled(bool)
            self.level1Prev.setEnabled(bool)
        else:
            self.level1Next.setEnabled(False)
            self.level1Prev.setEnabled(False)
        if label[2]:
            self.level2Next.setEnabled(bool)
            self.level2Prev.setEnabled(bool)
        else:
            self.level2Next.setEnabled(False)
            self.level2Prev.setEnabled(False)
        if label[3]:
            self.level3Next.setEnabled(bool)
            self.level3Prev.setEnabled(bool)
        else:
            self.level3Next.setEnabled(False)
            self.level3Prev.setEnabled(False)
        if label[4]:
            self.level4Next.setEnabled(bool)
            self.level4Prev.setEnabled(bool)
        else:
            self.level4Next.setEnabled(False)
            self.level4Prev.setEnabled(False)

    def updateParams(self):
        self.currentFrameList = self.frameList(self.currentPatient, self.currentSet)
        self.currentFrameIdx = 0
        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.frameOffset = 0

        self.currentDCMImgs, self.currentDCMPaths = self.getDicoms(self.currentPatient,
                                                                   self.currentFrame + self.frameOffset)
        self.currentDCMIdx = 0

        self.wiringPix = self.returnGuidewire(self.currentPatient, self.currentSet)

        self.currentLabel = self.returnLabels(self.currentPatient, self.currentSet, self.currentFrame)

        self.sliderUpdate()

        self.dcmInfoDisp()
        self.frameInfoDisp()
        self.setLevelButtons()


    def updateFrameNum(self):
        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.currentLabel = self.returnLabels(self.currentPatient, self.currentSet, self.currentFrame)
        self.currentDCMImgs, self.currentDCMPaths = self.getDicoms(self.currentPatient,
                                                                   self.currentFrame + self.frameOffset)


    def updateImage(self):
        pixmap = self.drawOverlay()
        self.overlay.setPixmap(pixmap)


    def drawOverlay(self):
        # showSel 0: DICOM 1: VG 2: Wiring 3: All label
        if self.showSel[0]:
            img = cv2.cvtColor(self.currentDCMImgs[self.currentDCMIdx], cv2.COLOR_GRAY2RGB)
        else:
            img = np.ones(self.imgSize, dtype=np.uint8) * 255
        label = self.currentLabel
        wire = self.wiringPix[self.currentWiringFrame]

        if self.showSel[2]:
            img = self.drawWire(wire, base=img, xyswap=True, value=[0, 255, 0])

        if self.showSel[1] and self.showSel[3]:
            img = self.drawAllLabels(label, base=img, xyswap=True, value=[255, 0, 0])
        elif self.showSel[1]:
            for key in label.keys():
                if self.levelShowSel[key]:
                    if label[key]:
                        img = self.drawLabels(label, key, self.levelSel[key], base=img, xyswap=True, value=self.levelColor[key])


        w, h, c = img.shape
        bytesPerLines = 3 * w
        qImg = QImage(img.data, w, h, bytesPerLines, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)
        return pixmap


    def drawWire(self, wire, base=None, size=(512,512), xyswap=False, value=[0, 255, 0]):
        if base is None:
            base = np.ones(size, dtype=np.uint8) * 255

        for x, y in wire:
            if xyswap:
                base[y, x] = value
            else:
                base[x, y] = value

        return base



    def showing(self):
        if self.showDICOM.isChecked(): self.showSel[0] = 1
        else: self.showSel[0] = 0

        if self.showVG.isChecked(): self.showSel[1] = 1
        else: self.showSel[1] = 0

        if self.showWiring.isChecked(): self.showSel[2] = 1
        else: self.showSel[2] = 0

        if self.showEvery.isChecked():
            self.showSel[3] = 1
            self.setLevelButtons()
        else:
            self.showSel[3] = 0
            self.setLevelButtons()

        self.updateImage()


    def setDCMOffsetUp(self):
        self.frameOffset += 1
        self.frameInfoDisp()
        self.updateFrameNum()
        self.updateImage()


    def setDCMOffsetDown(self):
        self.frameOffset -= 1
        self.frameInfoDisp()
        self.updateFrameNum()
        self.updateImage()


    def sliderUpdate(self):
        self.wiringSlider.setMaximum(len(self.wiringPix)-1)
        self.wiringSlider.setValue(0)
        self.updateImage()


    def sliderWire(self, value):
        self.currentWiringFrame = value

        self.frameInfoDisp()
        self.updateImage()


    def setWireNext(self):
        self.currentWiringFrame += 1
        if self.currentWiringFrame > len(self.wiringPix) - 1:
            self.currentWiringFrame = len(self.wiringPix) - 1

        self.frameInfoDisp()
        self.updateImage()

    def setWirePrev(self):
        self.currentWiringFrame -= 1
        if self.currentWiringFrame < 0:
            self.currentWiringFrame = 0

        self.frameInfoDisp()
        self.updateImage()


    def setAngioFrameNext(self):
        self.currentFrameIdx += 1
        if self.currentFrameIdx > len(self.currentFrameList) - 1:
            self.currentFrameIdx = len(self.currentFrameList) - 1

        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.frameInfoDisp()
        self.updateFrameNum()
        self.updateImage()

    def setAngioFramePrev(self):
        self.currentFrameIdx -= 1
        if self.currentFrameIdx < 0:
            self.currentFrameIdx = 0

        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.frameInfoDisp()
        self.updateFrameNum()
        self.updateImage()



    def frameInfoDisp(self):
        self.wiringFrameDisp.setText("{}".format(self.currentWiringFrame))
        self.angioFrameDisp.setText("{}".format(self.currentFrame))
        self.dcmOffsetDisp.setText("{}".format(self.frameOffset))

    def dcmInfoDisp(self):
        self.dcmNumDisp.setText("{}".format(self.currentDCMIdx))
        self.dcmPathDisp.setPlainText("{}".format(self.currentDCMPaths[self.currentDCMIdx]))


    def setPNext(self):
        self.currentPatientIdx += 1
        if self.currentPatientIdx > len(list(self.TotalList.keys())) - 1:
            self.currentPatientIdx = len(list(self.TotalList.keys())) - 1

        self.currentPatient = list(self.TotalList.keys())[self.currentPatientIdx]
        self.currentSetList = self.TotalList[self.currentPatient]
        self.currentSetIdx = 0
        self.currentSet = self.currentSetList[self.currentSetIdx]

        self.pnumDisp.setText("{}".format(self.currentPatient))
        self.psetDisp.setText("{}".format(self.currentSet))

        self.updateParams()
        self.updateImage()

    def setPPrev(self):
        self.currentPatientIdx -= 1
        if self.currentPatientIdx < 0:
            self.currentPatientIdx = 0

        self.currentPatient = list(self.TotalList.keys())[self.currentPatientIdx]
        self.currentSetList = self.TotalList[self.currentPatient]
        self.currentSetIdx = 0
        self.currentSet = self.currentSetList[self.currentSetIdx]

        self.pnumDisp.setText("{}".format(self.currentPatient))
        self.psetDisp.setText("{}".format(self.currentSet))

        self.updateParams()
        self.updateImage()


    def setPSNext(self):
        self.currentSetIdx += 1
        if self.currentSetIdx > len(self.currentSetList) - 1:
            self.currentSetIdx = len(self.currentSetList) - 1

        self.currentSet = self.currentSetList[self.currentSetIdx]
        self.wiringPix = self.returnGuidewire(self.currentPatient, self.currentSet)
        self.psetDisp.setText("{}".format(self.currentSet))
        self.updateFrameNum()
        self.updateImage()


    def setPSPrev(self):
        self.currentSetIdx -= 1
        if self.currentSetIdx < 0:
            self.currentSetIdx = 0

        self.currentSet = self.currentSetList[self.currentSetIdx]
        self.wiringPix = self.returnGuidewire(self.currentPatient, self.currentSet)
        self.psetDisp.setText("{}".format(self.currentSet))
        self.updateFrameNum()
        self.updateImage()



    def setDCMNumNext(self):
        self.currentDCMIdx += 1
        if self.currentDCMIdx > len(self.currentDCMPaths) - 1:
            self.currentDCMIdx = len(self.currentDCMPaths) - 1

        self.dcmInfoDisp()
        self.updateImage()

    def setDCMNumPrev(self):
        self.currentDCMIdx -= 1
        if self.currentDCMIdx < 0:
            self.currentDCMIdx = 0

        self.dcmInfoDisp()
        self.updateImage()


    def patientSet(self, root='./angio_labels' , file='**/*_level1.txt'):
        dirs = glob(os.path.join(root, file), recursive=True)
        dirs.sort()

        plists = {}

        for subpath in dirs:
            splitted = subpath.split('/')
            patient = int(splitted[2])
            if not (patient in plists.keys()):
                plists[patient] = []

            if len(splitted) == 5:
                patient = int(splitted[2])
                set = splitted[3]
                if not (set in plists[patient]):
                    plists[patient].append(set)

        return plists

    def setDirectory(self, p, set, root='./angio_labels'):
        patient = "{0:02d}".format(p)
        path = os.path.join(root, patient)
        path = os.path.join(path, set)
        return path

    def frameList(self, p, set, root='./angio_labels'):
        path = self.setDirectory(p, set, root)

        labels = glob(os.path.join(path, '*level1.txt'))
        labels.sort()

        fList = []

        for label in labels:
            frame = int(label.split('/')[-1].split('_')[0])
            fList.append(frame)

        return fList

    def returnLabels(self, p, set, frame):
        path = self.setDirectory(p, set)
        labels = {1: None, 2: None, 3: None, 4: None}

        for level in range(1,5):
            nbName = '{}_level{}_nb.txt'.format(frame, level)
            nbPath = os.path.join(path, nbName)
            pixPath = nbPath.replace('_nb', '')

            try:
                nbf = open(nbPath, 'r')
            except FileNotFoundError:
                continue
            numPix = []
            for line in nbf:
                numPix.append(int(float(line)))
            nbf.close()

            lines = []
            locf = open(pixPath, 'r')
            locData = locf.readlines()
            for i in range(len(numPix)):
                temp = []
                if i == 0:
                    startIdx = 0
                    endIdx = numPix[i]
                else:
                    startIdx = endIdx
                    endIdx = endIdx + numPix[i]
                for j in range(startIdx, endIdx):
                    xy = locData[j]
                    xstr = xy.split(' ')[0]
                    ystr = xy.split(' ')[1]
                    x = int(float(xstr))
                    y = int(float(ystr))
                    if x > 511: x = 511
                    if y > 511: y = 511
                    temp.append([x, y])
                lines.append(temp)

            labels[level] = lines

        return labels

    def drawLabels(self, labels, level=1, idx=0, base=None, size=(512, 512), xyswap=True, value=[255, 0, 0]):
        line = labels[level][idx]
        if base is None:
            base = np.ones(size, dtype=np.uint8) * 255

        for x, y in line:
            if xyswap:
                base[y, x] = value
            else:
                base[x, y] = value

        return base

    def drawAllLabels(self, labels, base=None, size=(512, 512), xyswap=True, value=0):
        if base is None:
            base = np.ones(size, dtype=np.uint8) * 255

        for key in labels.keys():
            if labels[key]:
                lineNum = len(labels[key])

                for i in range(lineNum):
                    line = labels[key][i]
                    for x, y in line:
                        if xyswap:
                            base[y, x] = value
                        else:
                            base[x, y] = value

        return base

    def getDicoms(self, patient, frame, root='./dicoms'):
        path = os.path.join(root, '{0:02d}'.format(patient))
        dpaths = glob(os.path.join(path, '**/IMG*'))
        dpaths.sort()

        imgs = []

        for dcmpath in dpaths:
            dcmdata = pydicom.dcmread(dcmpath)
            try:
                angios = dcmdata.pixel_array
                imgs.append(np.array(angios[frame]))
            except IndexError:
                imgs.append(np.ones((512, 512), dtype=np.uint8) * 255)
            except ValueError:
                imgs.append(np.ones((512, 512), dtype=np.uint8) * 255)

        return imgs, dpaths

    def returnGuidewire(self, p, set, drawCat=False):
        path = self.setDirectory(p, set, root='./gw_labels')
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


    def getIntersection(self):
        angio = self.patientSet(self.angioRoot)
        wiring = self.patientSet(self.wiringRoot, '**/*.txt')

        intersection = {}

        angioKeys = list(angio.keys())
        wiringKeys = list(wiring.keys())

        interKeys = list(set(angioKeys) & set(wiringKeys))

        for key in interKeys:
            angioSetList = angio[key]
            wiringSetList = wiring[key]
            interSetList = list(set(angioSetList) & set(wiringSetList))
            
            if interSetList:
                intersection[key] = interSetList
                
        return intersection








if __name__ == '__main__':
    app = QApplication(sys.argv)
    sel = LabelSelection()
    app.exec_()
    #plists = sel.patientSet('./gw_labels', '**/*.txt')
    #gwImgs = sel.drawGuidewires(1, 'set01')
    #labels = returnLabels(1, 'set01', 43)
    #test = drawLabels(labels, 'L1', 1)
    #test = drawAllLabels(labels)
