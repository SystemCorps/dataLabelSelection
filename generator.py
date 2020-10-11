import cv2
import pydicom
import numpy as np
import os
import sys
from glob import glob
import json
import pickle
import gzip
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
        self.parent = parent

        self.resultPrev.clicked.connect(self.parent.setVGSavedPrev)
        self.resultNext.clicked.connect(self.parent.setVGSavedNext)
        #self.updateResult()

    def updateResult(self):
        pixmap = self.drawResult()
        self.resultPixmap.setPixmap(pixmap)
        
    def drawResult(self):
        img = cv2.cvtColor(self.parent.currentDCMImgs[self.parent.currentDCMIdx], cv2.COLOR_GRAY2RGB)
        
        try:
            selectedLabel = self.parent.database[self.parent.currentPatient][self.parent.currentSet][self.parent.currentFrame]['SelectedLabels']
            if selectedLabel:
                line = self.parent.makeSingleLine(self.parent.currentPatient,
                                                  self.parent.currentSet, self.parent.currentFrame,
                                                  self.parent.showSavedIdx)
            
                img = self.parent.drawSavedLabel(line, img, xyswap=True, value=[255, 0, 0])

        except (KeyError, TypeError):
            pass
        
        w, h, c = img.shape
        bytesPerLines = 3 * w
        qImg = QImage(img.data, w, h, bytesPerLines, QImage.Format_RGB888)
        pixmap = QPixmap(qImg)

        return pixmap



class LabelSelection(QMainWindow):

    def __init__(self, uiPath='mainwindow.ui', angioRoot='.'+os.sep+'angio_labels', wiringRoot='.'+os.sep+'gw_labels', dcmRoot='.'+os.sep+'dicoms', imgSize=(512,512, 3)):
        super(LabelSelection, self).__init__()
        uic.loadUi(uiPath, self)
        self.show()

        try:
            self.savePath = QFileDialog.getSaveFileName(self, 'Save File')[0]
        except:
            self.savePath = "setting.pkl"
        self.savePathDisp.setText(self.savePath)
        self.genSavePath = None

        self.angioRoot = angioRoot
        self.wiringRoot = wiringRoot
        self.dcmRoot = dcmRoot
        self.imgSize = imgSize

        self.steps = 50
        self.minLen = 30

        self.TotalList = self.getIntersection()

        try:
            with open(self.savePath, 'rb') as pklfile:
                self.database = pickle.load(pklfile)
            self.database.keys()
        except:
            self.database = dict.fromkeys(list(self.TotalList.keys()))

        self.ResultWindow = ResultWindow(self)


        # Selectable
        self.showSel = [1, 0, 0, 1]
        self.levelShowSel = {1:0, 2:0, 3:0, 4:0}
        self.levelSel = {1:0, 2:0, 3:0, 4:0}
        self.levelColor = {1:[255, 0, 0], 2:[255, 204, 51], 3:[51, 255, 255], 4:[0, 0, 255]}
        self.showSavedIdx = 0
        self.showSavedFlag = 0

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
        try:
            dcmPath = self.database[self.currentPatient][self.currentSet]['DCMPath']
            for i in range(len(self.currentDCMPaths)):
                if dcmPath == self.currentDCMPaths[i]:
                    self.currentDCMIdx = i
        except:
             self.currentDCMIdx = 0

        self.sliderUpdate()


        self.pnumDisp.setText("{}".format(self.currentPatient))
        self.psetDisp.setText("{}".format(self.currentSet))

        # Buttons
        self.dcmNumNext.clicked.connect(self.setDCMNumNext)
        self.dcmNumPrev.clicked.connect(self.setDCMNumPrev)
        self.genData.clicked.connect(self.generateDataset)
        self.genPath.clicked.connect(self.setGenSaveRoot)
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
        self.delVG.clicked.connect(self.delVGset)
        self.savePathBut.clicked.connect(self.setSavePath)
        self.saveSettingBut.clicked.connect(self.save)
        self.setDICOM.clicked.connect(self.setDCMset)
        self.setVG.clicked.connect(self.setVGset)
        self.wiringNext.clicked.connect(self.setWireNext)
        self.wiringPrev.clicked.connect(self.setWirePrev)
        self.angioFrameNext.clicked.connect(self.setAngioFrameNext)
        self.angioFramePrev.clicked.connect(self.setAngioFramePrev)
        self.dcmOffsetUp.clicked.connect(self.setDCMOffsetUp)
        self.dcmOffsetDown.clicked.connect(self.setDCMOffsetDown)
        self.showResult.clicked.connect(self.showResultWindow)

        self.setSavedPrev.clicked.connect(self.setVGSavedPrev)
        self.setSavedNext.clicked.connect(self.setVGSavedNext)


        # checkbox
        self.showDICOM.stateChanged.connect(self.showing)
        self.showVG.stateChanged.connect(self.showing)
        self.showWiring.stateChanged.connect(self.showing)
        self.showEvery.stateChanged.connect(self.showing)
        self.level1Check.stateChanged.connect(self.levelCheck)
        self.level2Check.stateChanged.connect(self.levelCheck)
        self.level3Check.stateChanged.connect(self.levelCheck)
        self.level4Check.stateChanged.connect(self.levelCheck)
        self.showVGSaved.stateChanged.connect(self.showSaved)
        # slider
        self.wiringSlider.valueChanged[int].connect(self.sliderWire)

        # Display
        self.dcmInfoDisp()
        self.frameInfoDisp()
        self.levelInfoDisp()
        self.showSavedEnb()
        # Image Show
        self.updateImage()

    def closeEvent(self, event):
        self.save()
    
    def setSavePath(self):
        self.savePath = QFileDialog.getSaveFileName(self, 'Save File', "setting.txt")
        self.savePathDisp.setText(self.savePath)

    def save(self):
        with open(self.savePath, 'wb') as pklfile:
            pickle.dump(self.database, pklfile, pickle.HIGHEST_PROTOCOL)

    def setGenSaveRoot(self):
        self.genSavePath = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.genPathDisp.setText(self.genSavePath)


    def generateDataset(self):
        if not self.genSavePath:
            self.setGenSaveRoot()

        for patient in self.database.keys():
            if self.database[patient]:
                pData = self.database[patient]
                for set in pData.keys():
                    sData = pData[set]
                    dcmData = pydicom.dcmread(sData['DCMPath'])
                    angios = dcmData.pixel_array
                    
                    frameList = list(sData.keys())
                    frameList.remove('DCMPath')

                    for frame in frameList:
                        refAngio = angios[frame]
                        npAngio = np.array(refAngio, dtype=np.uint8)
                        PILAngio = Image.fromarray(npAngio)
                        head, name, full = self.savePathName(patient, set, 'Angio', frame, angio=True)
                        if not os.path.exists(head):
                            os.makedirs(head)
                        PILAngio.save(full)

                        selectedLabels = sData[frame]['SelectedLabels']

                        for i in range(len(selectedLabels)):
                            singleLine = self.makeSingleLine(patient, set, frame, idx=i)
                            splitted = self.genSplitted(singleLine)

                            for j in range(self.steps):
                                lineImg = self.drawSavedLabel(splitted[j], xyswap=True, value=0)
                                PILImg = Image.fromarray(lineImg)
                                head, name, full = self.savePathName(patient, set, 'virtualGW', frame, subIdx=i, stepIdx=j)
                                if not os.path.exists(head):
                                    os.makedirs(head)
                                PILImg.save(full)

                                npLine = np.array(lineImg, dtype=np.uint8)

                                regImg = ((npAngio/255) * (npLine/255) * 255).astype(np.uint8)
                                regPILImg = Image.fromarray(regImg)
                                head, name, full = self.savePathName(patient, set, 'virtualRG', frame, subIdx=i, stepIdx=j)
                                if not os.path.exists(head):
                                    os.makedirs(head)
                                regPILImg.save(full)
                        


    def savePathName(self, patient, set, folder, frame, subIdx=None, stepIdx=None, angio=False):
        temp = os.path.join(self.genSavePath, "{0:02d}".format(patient))
        temp = os.path.join(temp, set)
        temp = os.path.join(temp, folder)
        if subIdx is not None:
            temp = os.path.join(temp, "{0:02d}".format(subIdx))
        head = temp
        if angio:
            name = "{0:05d}.png".format(frame)
        else:
            name = "{0:03d}_".format(frame) + "{0:05d}.png".format(stepIdx)
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
            

                            

    def showResultWindow(self):
        #ResultWindow(self)
        #self.ResultWindow()
        self.ResultWindow.show()
        self.ResultWindow.updateResult()


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



    def delVGset(self):
        try:
            del self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels'][self.showSavedIdx]
        except:
            pass

        self.showSavedIdx -= 1
        if self.showSavedIdx < 0:
            self.showSavedIdx = 0

        self.showSavedDisp()
        self.updateImage()

    def setVGSavedPrev(self):
        self.showSavedIdx -= 1
        if self.showSavedIdx < 0:
            self.showSavedIdx = 0

        self.showSavedDisp()
        self.updateImage()


    def setVGSavedNext(self):
        self.showSavedIdx += 1
        try:
            selectedLabels = self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels']
            numLabels = len(selectedLabels)
        except:
            numLabels = 0

        if self.showSavedIdx > numLabels-1:
            self.showSavedIdx = numLabels-1
        elif numLabels == 0:
            self.showSavedIdx = 0       

        if numLabels:
            text = "{}/{}".format(self.showSavedIdx + 1, numLabels)
            self.selectedSaved.setText(text)
        else:
            self.selectedSaved.setText("None")

        self.updateImage()


    def showSaved(self):
        if self.showVGSaved.isChecked():
            self.showSavedFlag = True
            self.updateImage()
        else:
            self.showSavedFlag = False
            self.updateImage()


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


    def showSavedDisp(self):
        try:
            selectedLabels = self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels']
            numLabels = len(selectedLabels)
        except:
            numLabels = 0

        if numLabels:
            text = "{}/{}".format(self.showSavedIdx + 1, numLabels)
            self.selectedSaved.setText(text)
        else:
            self.selectedSaved.setText("None")
            self.showSavedFlag = False
            self.showVGSaved.setEnabled(False)
            self.showVGSaved.setChecked(False)


    def showSavedEnb(self):
        try:
            selectedLabels = self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels'][self.showSavedIdx]
            self.showVGSaved.setEnabled(True)
        except:
            self.showVGSaved.setEnabled(False)
        self.showSavedDisp()
    

    def setVGset(self):
        if self.currentFrame in self.database[self.currentPatient][self.currentSet].keys():
            selectedLabels = {}
            for key in self.levelShowSel.keys():
                if self.levelShowSel[key]:
                    selectedLabels[key] = self.currentLabel[key][self.levelSel[key]]

            self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels'].append(selectedLabels)
        else:
            self.database[self.currentPatient][self.currentSet][self.currentFrame] = {}
            self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels'] = []
            
            selectedLabels = {}
            for key in self.levelShowSel.keys():
                if self.levelShowSel[key]:
                    selectedLabels[key] = self.currentLabel[key][self.levelSel[key]]

            self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels'].append(selectedLabels)
        self.showVGSaved.setEnabled(True)
        self.showSavedDisp()
        self.updateImage()
        self.save()
    

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
        self.updateImage()


    def updateImage(self):
        pixmap = self.drawOverlay()
        self.overlay.setPixmap(pixmap)
        self.ResultWindow.updateResult()



    def drawOverlay(self):
        # showSel 0: DICOM 1: VG 2: Wiring 3: All label
        if self.showSavedFlag:
            selectedLabels = self.database[self.currentPatient][self.currentSet][self.currentFrame]['SelectedLabels'][self.showSavedIdx]
            dcmPath = self.database[self.currentPatient][self.currentSet]['DCMPath']
            dcm = self.getSingleFrameDicom(dcmPath, self.currentFrame)
            img = cv2.cvtColor(dcm, cv2.COLOR_GRAY2RGB)
            for key in selectedLabels.keys():
                label = selectedLabels[key]
                img = self.drawSavedLabel(label, img, xyswap=True)

        else:
            self.showVGSaved.setChecked(False)
            if self.showSel[0]:
                try:
                    dcmPath = self.database[self.currentPatient][self.currentSet]['DCMPath']
                    dcm = self.getSingleFrameDicom(dcmPath, self.currentFrame)
                    img = cv2.cvtColor(dcm, cv2.COLOR_GRAY2RGB)
                except:
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
        self.currentWiringFrame = 0
        self.showSavedFlag = 0
        self.showSavedIdx = 0
        self.showVGSaved.setChecked(False)
        self.currentFrameList = self.frameList(self.currentPatient, self.currentSet)
        self.currentFrameIdx = 0
        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.frameOffset = 0
        self.levelShowSel = {1:0, 2:0, 3:0, 4:0}
        self.levelSel = {1:0, 2:0, 3:0, 4:0}
        self.level1Check.setChecked(False)
        self.level2Check.setChecked(False)
        self.level3Check.setChecked(False)
        self.level4Check.setChecked(False)
        self.currentDCMImgs, self.currentDCMPaths = self.getDicoms(self.currentPatient,
                                                                   self.currentFrame + self.frameOffset)
        
        try:
            dcmPath = self.database[self.currentPatient][self.currentSet]['DCMPath']
            for i in range(len(self.currentDCMPaths)):
                if dcmPath == self.currentDCMPaths[i]:
                    self.currentDCMIdx = i
        except:
            self.currentDCMIdx = 0

        self.wiringPix = self.returnGuidewire(self.currentPatient, self.currentSet)

        self.currentLabel = self.returnLabels(self.currentPatient, self.currentSet, self.currentFrame)

        self.sliderUpdate()

        self.dcmInfoDisp()
        self.frameInfoDisp()
        self.setLevelButtons()
        self.showSavedEnb()


    def updateFrameNum(self):
        #self.currentWiringFrame = 0
        self.showSavedFlag = 0
        self.showSavedIdx = 0
        self.showVGSaved.setChecked(False)
        self.levelShowSel = {1:0, 2:0, 3:0, 4:0}
        self.levelSel = {1:0, 2:0, 3:0, 4:0}
        self.level1Check.setChecked(False)
        self.level2Check.setChecked(False)
        self.level3Check.setChecked(False)
        self.level4Check.setChecked(False)
        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.currentLabel = self.returnLabels(self.currentPatient, self.currentSet, self.currentFrame)
        self.currentDCMImgs, self.currentDCMPaths = self.getDicoms(self.currentPatient,
                                                                   self.currentFrame + self.frameOffset)
        
        self.showSavedEnb()


   


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
        self.wiringSlider.setValue(self.currentWiringFrame)
        self.updateImage()


    def sliderWire(self, value):
        self.currentWiringFrame = value

        self.frameInfoDisp()
        self.updateImage()


    def setWireNext(self):
        self.currentWiringFrame += 1
        if self.currentWiringFrame > len(self.wiringPix) - 1:
            self.currentWiringFrame = len(self.wiringPix) - 1

        self.sliderUpdate()
        self.frameInfoDisp()
        self.updateImage()

    def setWirePrev(self):
        self.currentWiringFrame -= 1
        if self.currentWiringFrame < 0:
            self.currentWiringFrame = 0

        self.sliderUpdate()
        self.frameInfoDisp()
        self.updateImage()


    def setAngioFrameNext(self):
        self.currentFrameIdx += 1
        if self.currentFrameIdx > len(self.currentFrameList) - 1:
            self.currentFrameIdx = len(self.currentFrameList) - 1

        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.frameInfoDisp()
        self.updateFrameNum()
        self.levelInfoDisp()
        self.updateImage()

    def setAngioFramePrev(self):
        self.currentFrameIdx -= 1
        if self.currentFrameIdx < 0:
            self.currentFrameIdx = 0

        self.currentFrame = self.currentFrameList[self.currentFrameIdx]
        self.frameInfoDisp()
        self.updateFrameNum()
        self.levelInfoDisp()
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

        try:
            dcmPath = self.database[self.currentPatient][self.currentSet]['DCMPath']
            for i in range(len(self.currentDCMPaths)):
                if dcmPath == self.currentDCMPaths[i]:
                    self.currentDCMIdx = i
        except:
             self.currentDCMIdx = 0
        
        self.wiringPix = self.returnGuidewire(self.currentPatient, self.currentSet)
        self.psetDisp.setText("{}".format(self.currentSet))
        self.currentFrameList = self.frameList(self.currentPatient, self.currentSet)
        self.currentFrameIdx = 0
        self.updateFrameNum()
        self.frameInfoDisp()
        self.currentWiringFrame = 0
        self.sliderUpdate()
        self.updateImage()


    def setPSPrev(self):
        self.currentSetIdx -= 1
        if self.currentSetIdx < 0:
            self.currentSetIdx = 0

        self.currentSet = self.currentSetList[self.currentSetIdx]
        
        try:
            dcmPath = self.database[self.currentPatient][self.currentSet]['DCMPath']
            for i in range(len(self.currentDCMPaths)):
                if dcmPath == self.currentDCMPaths[i]:
                    self.currentDCMIdx = i
        except:
             self.currentDCMIdx = 0
        
        self.wiringPix = self.returnGuidewire(self.currentPatient, self.currentSet)
        self.psetDisp.setText("{}".format(self.currentSet))
        self.currentFrameList = self.frameList(self.currentPatient, self.currentSet)
        self.currentFrameIdx = 0
        self.updateFrameNum()
        self.frameInfoDisp()
        self.currentWiringFrame = 0
        self.sliderUpdate()
        self.updateImage()



    def setDCMNumNext(self):
        try:
            dcmPath = self.database[self.currentPatient][self.currentSet]['DCMPath']
            for i in range(len(self.currentDCMPaths)):
                if dcmPath == self.currentDCMPaths[i]:
                    self.currentDCMIdx = i
        except:
            self.currentDCMIdx += 1
        if self.currentDCMIdx > len(self.currentDCMPaths) - 1:
            self.currentDCMIdx = len(self.currentDCMPaths) - 1

        self.dcmInfoDisp()
        self.updateImage()

    def setDCMNumPrev(self):
        try:
            dcmPath = self.database[self.currentPatient][self.currentSet]['DCMPath']
            for i in range(len(self.currentDCMPaths)):
                if dcmPath == self.currentDCMPaths[i]:
                    self.currentDCMIdx = i
        except:
            self.currentDCMIdx -= 1
        if self.currentDCMIdx < 0:
            self.currentDCMIdx = 0

        self.dcmInfoDisp()
        self.updateImage()


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

    def setDirectory(self, p, set, root='.'+os.sep+'angio_labels'):
        patient = "{0:02d}".format(p)
        path = os.path.join(root, patient)
        path = os.path.join(path, set)
        return path

    def frameList(self, p, set, root='.'+os.sep+'angio_labels'):
        path = self.setDirectory(p, set, root)

        labels = glob(os.path.join(path, '*level1.txt'))
        labels.sort()

        fList = []

        for label in labels:
            frame = int(label.split(os.sep)[-1].split('_')[0])
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

                if len(lines):
                    lastPos = np.array(lines[-1][-1])
                    curStart = np.array(temp[0])
                    dist1 = np.linalg.norm(lastPos-curStart)

                    curEnd = np.array(temp[-1])
                    lastStart = np.array(lines[-1][0])
                    dist2 = np.linalg.norm(curEnd-lastStart)

                    if (dist1 < 10) or (dist2 < 10):
                        lines[-1].extend(temp)
                    else:
                        lines.append(temp)
                else:
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

    def drawAllLabels(self, labels, base=None, size=(512, 512, 3), xyswap=True, value=0):
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


    def getSingleFrameDicom(self, path, frame):
        dcmdata = pydicom.dcmread(path)
        try:
            angios = dcmdata.pixel_array
            img = np.array(angios[frame])
        except IndexError:
            img = np.ones((512, 512), dtype=np.uint8) * 255
        except ValueError:
            img = np.ones((512, 512), dtype=np.uint8) * 255
        return img


    def getDicoms(self, patient, frame, root='.'+os.sep+'dicoms'):
        path = os.path.join(root, '{0:02d}'.format(patient))
        dpaths = glob(os.path.join(path, '**'+os.sep+'IMG*'))
        dpaths.sort()

        imgs = []
        validPaths = []

        for dcmpath in dpaths:
            dcmdata = pydicom.dcmread(dcmpath)
            try:
                angios = dcmdata.pixel_array
                if len(angios[frame].shape) < 2:
                    continue
                imgs.append(np.array(angios[frame]))
                validPaths.append(dcmpath)
            except:
                pass


        return imgs, validPaths

    def returnGuidewire(self, p, set, drawCat=False):
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
        wiring = self.patientSet(self.wiringRoot, '**'+os.sep+'*.txt')

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
    #plists = sel.patientSet('.'+os.sep+'gw_labels', '**'+os.sep+'*.txt')
    #gwImgs = sel.drawGuidewires(1, 'set01')
    #labels = returnLabels(1, 'set01', 43)
    #test = drawLabels(labels, 'L1', 1)
    #test = drawAllLabels(labels)
