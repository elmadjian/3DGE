import sys
from PySide2.QtGui import QGuiApplication
from PySide2.QtQml import QQmlApplicationEngine, qmlRegisterType
from PySide2.QtCore import QUrl, Property, Signal, QObject, Slot
import eye
import videoio_uvc
import calibration_hmd
import cv2
import time
import numpy as np
import multiprocessing as mp


if __name__=='__main__':
    #mp.set_start_method('spawn')
    app = QGuiApplication(sys.argv)
    app.setOrganizationName('Cadu')
    app.setOrganizationDomain('Nowhere')
    engine = QQmlApplicationEngine()
    
    videoio   = videoio_uvc.VideoIO_UVC()
    calib_hmd = calibration_hmd.HMDCalibrator(3, 3, 21, 3) 

    le_cam    = eye.EyeCamera('left')
    re_cam    = eye.EyeCamera('right')
    videoio.set_active_cameras(le_cam, re_cam)
    calib_hmd.set_sources(le_cam, re_cam)

    engine.rootContext().setContextProperty("camManager", videoio)
    engine.rootContext().setContextProperty("leftEyeCam", le_cam)
    engine.rootContext().setContextProperty("rightEyeCam", re_cam)
    engine.rootContext().setContextProperty("calibHMD", calib_hmd)
    engine.addImageProvider('leyeimg', le_cam)
    engine.addImageProvider('reyeimg', re_cam)
    engine.load(QUrl("../UI/qml/main.qml"))


    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())