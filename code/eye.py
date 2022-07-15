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
#from pupil_detectors import Detector3D, Roi
from pupil_detectors import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode,\
                              Prediction



class EyeCamera(camera.Camera):

    def __init__(self, name=None, mode=(192,192,120)):
        super().__init__(name)
        self.mode = mode
        self.cam_process = None
        self.vid_process = None
        self.recorder_process = None
        self.shared_array = self.create_shared_array(mode)
        camera = CameraModel(focal_length=561.5, resolution=[mode[0], mode[1]])
        self.detector_2d = Detector2D()
        self.detector_3d = Detector3D(camera=camera, 
               long_term_mode=DetectorMode.blocking)
        self.pos = None
        #self.detector.update_properties({'2d':{'pupil_size_max':180}})
        #self.detector.update_properties({'2d':{'pupil_size_min':10}})
        self.countdown = 5
        self.update = True

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
        result_2d = self.detector_2d.detect(gray)
        result_2d['timestamp'] = timestamp
        result = self.update_and_detect(result_2d, gray, update=self.update)
        if result["model_confidence"] > 0.5:
            sphere = result["projected_sphere"]
            self.__draw_ellipse(sphere, img, (255,120,120), 1) 
        if result["confidence"] > 0.7:
            n = np.array(result['circle_3d']['normal']) 
            self.__draw_tracking_info(result, img)
            self.pos = np.array([n[0], n[1], n[2], time.monotonic()])
            self.countdown = 5
        else:
            self.countdown -= 1
            if self.countdown <= 0:
                self.pos = None
        return img


    def update_and_detect(self, pupil_datum, frame, 
           apply_refraction_correction=True, update=True):
        observation = self.detector_3d._extract_observation(pupil_datum)
        if update:
            self.detector_3d.update_models(observation)
        sphere_center = self.detector_3d.long_term_model.sphere_center
        pupil_circle = self.detector_3d._predict_pupil_circle(observation, frame)
        prediction_uncorrected = Prediction(sphere_center, pupil_circle)
        if apply_refraction_correction:
            pupil_circle = self.detector_3d.long_term_model.apply_refraction_correction(
                pupil_circle
            )
            sphere_center = self.detector_3d.long_term_model.corrected_sphere_center
        prediction_corrected = Prediction(sphere_center, pupil_circle)
        result = self.detector_3d._prepare_result(
            observation,
            prediction_uncorrected=prediction_uncorrected,
            prediction_corrected=prediction_corrected,
        )
        return result

    def freeze_model(self):
        self.update = False

    def unfreeze_model(self):
        self.update = True
    
    def __draw_tracking_info(self, result, img, color=(255,120,120)):
        ellipse = result["ellipse"]
        normal = result["circle_3d"]["normal"]
        center = tuple(int(v) for v in ellipse["center"])
        cv2.drawMarker(img, center, (0,255,0), cv2.MARKER_CROSS, 12, 1)
        self.__draw_ellipse(ellipse, img, (0,0,255))
        dest_pos = (int(center[0]+normal[0]*60), int(center[1]+normal[1]*60))
        cv2.line(img, center, dest_pos, (85,175,20),2)    


    def __draw_ellipse(self, ellipse, img, color, thickness=2):
        center = tuple(int(v) for v in ellipse["center"])
        axes = tuple(int(v/2) for v in ellipse["axes"])
        rad = ellipse["angle"]
        cv2.ellipse(img, center, axes, rad, 0, 360, color, 2)


    def reset_model(self):
        self.unfreeze_model()
        self.detector_3d.reset()


    def get_processed_data(self):
        return self.pos
        




            

