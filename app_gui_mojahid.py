
#% pip install PyQtChart


import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import random

from PyQt5 import uic, QtWidgets, sip
from PyQt5.QtWidgets import*
from PySide2.QtCharts import QtCharts

from matplotlib.backends.backend_qt5agg import *

from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure


qtCreateFile = './mainwindow.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreateFile)

#The main window - load the ui created by QT creator
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.pushButton_loading.clicked.connect(self.getCSV)
        #self.pushButton_plot.clicked.connect(self.plot)
        self.pushButton_EDA.clicked.connect(self.comboPlot)
        self.pushButton_feature.clicked.connect(self.random)

    def comboPlot(self):
        if self.comboBox_eda.currentText()=='Severity':
            self.comboBox_eda.currentIndexChanged.connect(self.severity)
        elif self.comboBox_eda.currentText()=='Temperature':
            self.comboBox_eda.currentIndexChanged.connect(self.temperature)
        else :
            self.MplWidget.canvas.axes.clear()


    def getCSV(self):
            self.df=pd.read_csv('./us_data_combined.csv')
            stat_st = 'Data Description:'+'\n'+'\n' + str(self.df.describe())
            sample = 'Sample Data:' + '\n' + '\n' + str(self.df.head(5))
            self.textEdit_output.setText(stat_st)
            self.textEdit_output.setText(stat_st + '\n' + '\n' + sample)
            self.progressBar.setValue(100)

    def severity(self):
        labels = ['High','Low']
        severity_count = self.df.groupby("Severity")["Severity"].count()
        explode = (0, 0)

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.pie(severity_count, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True)
        #self.MplWidget.canvas.axes.legend(('High', 'Low'), loc='lower')
        self.MplWidget.canvas.axes.set_title('Severity Count ')
        self.MplWidget.canvas.draw()

    def temperature(self):
        temp= self.df["Temperature(F)"].dropna()

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.hist(temp, color='blue', edgecolor='black',bins=int(180 / 5))
        # self.MplWidget.canvas.axes.legend(('Temperature'), loc='upper right')
        self.MplWidget.canvas.axes.set_title('Temperature')
        self.MplWidget.canvas.draw()


    def random(self):
        labels = ['High','Low']
        severity_count = self.df.groupby("Severity")["Severity"].count()
        explode = (0, 0)

        #self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.pie(severity_count, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True)
        #self.MplWidget_2.canvas.axes.legend(('High', 'Low'), loc='lower')
        self.MplWidget_2.canvas.axes.set_title('Severity Count ')
        self.MplWidget_2.canvas.draw()
        self.MplWidget_2.canvas.update()
        self.tab_eda.setObjectName.active()

if __name__=="__main__":
    app= QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())