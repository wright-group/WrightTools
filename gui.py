'''
User interface for WrightTools.
'''

### import ####################################################################


from __future__ import absolute_import, division, print_function, unicode_literals

import sys

from PyQt4 import QtGui, QtCore
app = QtGui.QApplication(sys.argv)

from . import __version__


### MainWindow ################################################################


class MainWindow(QtGui.QMainWindow):
    
    def __init__(self):        
        QtGui.QMainWindow.__init__(self, parent=None)        
        self.setWindowTitle('WrightTools')
        self.window_verti_size = 600
        self.window_horiz_size = 600
        self.setGeometry(0,0, self.window_horiz_size, self.window_verti_size)
        self._center()
        self.resize(self.window_horiz_size, self.window_verti_size)
        # create own frame
        self.main_frame = QtGui.QWidget()        
        self.main_frame.setLayout(QtGui.QHBoxLayout())
        self.setCentralWidget(self.main_frame)
        # initialize GUI object
        self.gui = GUI(self.main_frame)
        
    def _center(self):
        '''
        Place window in center of screen.
        '''
        screen = QtGui.QDesktopWidget().screenGeometry() 
        size = self.geometry() 
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)

        

### GUI #######################################################################

        
class GUI(QtCore.QObject):
    
    def __init__(self, frame_widget):
        '''
        The actuall GUI.
        '''
        self.frame_widget = frame_widget
        self.frame_layout = frame_widget.layout()
        self.create_main_frame()
    
    def create_main_frame(self):
        label = QtGui.QLabel('test 12')
        self.frame_layout.addWidget(label)
    
    
### main ######################################################################


def main():
    global MainWindow
    MainWindow = MainWindow()
    MainWindow.show()
    MainWindow.showMaximized()
    app.exec_()
    return MainWindow

main_form = main()
