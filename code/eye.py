import cv2
import time
import numpy as np
import camera_proc as camera
import sys 
from matplotlib import pyplot as plt
#from eye_img_processor import EyeImageProcessor
from img_processor import ImageProcessor
from multiprocessing import Array, Process
import ctypes
import uvc
from pupil_detectors import Detector3D, Roi


class EyeCamera(camera.Camera):

    def __init__(self, name=None, mode=(192,192,120)):
        super().__init__(name)
        self.mode = mode
        self.cam_process = None
        self.vid_process = None
        self.shared_array = self.create_shared_array(mode)
        self.detector = Detector3D()
        self.bbox = None
        self.pos = None
        
        self.detector.update_properties({'2d':{'pupil_size_max':180}})
        self.detector.update_properties({'2d':{'pupil_size_min':10}})
        self.countdown = 5

    def init_process(self, source, pipe, array, mode, cap):
        mode = self.check_mode_availability(source, mode)
        self.cam_process = ImageProcessor(source, mode, pipe, array, cap)
        self.cam_process.start() 

    def init_vid_process(self, source, pipe, array, mode, cap):
        mode = self.check_mode_availability(source, mode)
        self.cam_process = ImageProcessor(source, mode, pipe, array, cap)
        self.vid_process = Process(target=self.cam_process.run_vid, args=())
        self.vid_process.start()

    def join_process(self):
        self.cam_process.join(10)

    def join_vid_process(self):
        self.vid_process.join(3)

    def create_shared_array(self, mode):
        w = mode[0]
        h = mode[1]
        return Array(ctypes.c_uint8, h*w*3, lock=False)

    def check_mode_availability(self, source, mode):
        dev_list = uvc.device_list()
        cap = uvc.Capture(dev_list[source]['uid'])
        if mode not in cap.avaible_modes:
            m = cap.avaible_modes[0]
            mode = (m[1], m[0], m[2])
            self.shared_array = self.create_shared_array(mode)
            self.mode = mode
        return mode

    def process(self, img):
        if img is None:
            return
        height, width = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        timestamp = uvc.get_time_monotonic()
        roi = None
        if self.bbox is not None:
            xmin, ymin, w, h = self.bbox
            roi = Roi(xmin, ymin, xmin+w, ymin+h)
        result = self.detector.detect(gray, timestamp, roi=roi)
        #print(result)
        if result["model_confidence"] > 0.25:
            sphere = result["projected_sphere"]
            self.__draw_ellipse(sphere, img, (255,120,120), 1) 
        if result["confidence"] > 0.5:
            n = np.array(result['circle_3d']['normal']) 
            self.bbox = self.__get_bbox(result, img)    
            self.__draw_tracking_info(result, img)
            # cv2.imshow("testando", img)
            # cv2.waitKey(1)
            self.pos = np.array([n[0], n[1], n[2], time.monotonic()])
            self.countdown = 5
        else:
            self.countdown -= 1
            if self.countdown <= 0:
                self.pos = None
                self.bbox = None
        return img

    def freeze_model(self):
         self.detector.update_properties({
             "3d": {"model_is_frozen": True}
             })

    def unfreeze_model(self):
        self.detector.update_properties({
             "3d": {"model_is_frozen": False}
             })

    def __get_bbox(self, result, img):
        r = result['diameter']
        point = result['ellipse']['center']
        x1 = point[0]-r*0.8
        y1 = point[1]-r*0.8
        x2 = point[0]+r*0.8
        y2 = point[1]+r*0.8
        x1 = self.__test_boundaries(x1, img.shape[1])
        y1 = self.__test_boundaries(y1, img.shape[0])
        x2 = self.__test_boundaries(x2, img.shape[1])
        y2 = self.__test_boundaries(y2, img.shape[0])
        w = x2-x1
        h = y2-y1
        cv2.rectangle(img, self.bbox, (125,80,80), 2, 1)
        return int(x1),int(y1),int(w),int(h)


    def __test_boundaries(self, x, lim):
        if x < 0:
            return 0
        if x >= lim:
            return lim-1
        return x


    
    def __draw_tracking_info(self, result, img, color=(255,120,120)):
        ellipse = result["ellipse"]
        normal = result["circle_3d"]["normal"]
        center = tuple(int(v) for v in ellipse["center"])
        cv2.drawMarker(img, center, (0,255,0), cv2.MARKER_CROSS, 12, 1)
        self.__draw_ellipse(ellipse, img, (0,0,255))
        dest_pos = (int(center[0]+normal[0]*60), int(center[1]+normal[1]*60))
        cv2.line(img, center, dest_pos, (85,175,20),2)
        # if self.bbox is not None:
        #     cv2.rectangle(img, self.bbox, (120,255,120), 2, 1)
        


    def __draw_ellipse(self, ellipse, img, color, thickness=2):
        center = tuple(int(v) for v in ellipse["center"])
        axes = tuple(int(v/2) for v in ellipse["axes"])
        rad = ellipse["angle"]
        cv2.ellipse(img, center, axes, rad, 0, 360, color, 2)


    def reset_model(self):
        self.detector.reset_model()


    def get_processed_data(self):
        return self.pos
        




            

