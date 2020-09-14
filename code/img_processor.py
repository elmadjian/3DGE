import uvc
from multiprocessing import Process, Pipe, Array
import traceback
import cv2
import sys
import time
import numpy as np
import ctypes
import traceback


class ImageProcessor(Process):

    def __init__(self, source, mode, pipe, array, cap):
        Process.__init__(self)
        self.eye_cam = False
        self.source = source
        self.mode = mode
        self.pipe = pipe
        self.shared_array = array
        self.capturing = cap
    
    def __get_shared_np_array(self, img):
        nparray = np.frombuffer(self.shared_array, dtype=ctypes.c_uint8) 
        return nparray.reshape(img.shape)

    def __adjust_gamma(self, img, gamma):
        lut = np.empty((1,256), np.uint8)
        for i in range(256):
            lut[0,i] = np.clip(pow(i/255.0, gamma) * 255.0, 0, 255)
        return cv2.LUT(img, lut)

    def __cvtBlackWhite(self, img, color):
        if color:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def __flip_img(self, img, flip):
        if flip:
            return cv2.flip(img, -1)
        return img

    def __reset_mode(self, cap):
        print("resetting...")
        mode = cap.frame_mode
        cap.close()
        time.sleep(0.5)
        dev_list = uvc.device_list()
        cap2 = uvc.Capture(dev_list[self.source]['uid'])
        print("Trying mode:", mode)
        cap2.frame_mode = mode
        cap2.bandwidth_factor = 1.3
        return cap2

    def __setup_eye_cam(self, cap):
        if self.eye_cam:
            try:
                controls_dict = dict([(c.display_name, c) for c in cap.controls])
                controls_dict['Auto Exposure Mode'].value = 1
                controls_dict['Gamma'].value = 200
            except:
                print("Exposure settings not available for this camera.")

    def run_vid(self):
        self.capturing.value = 1
        cap = cv2.VideoCapture(self.source)
        gamma, color, delay, flip = 1, True, 1/self.mode[2], False
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                img = self.__adjust_gamma(frame, gamma)
                img = self.__cvtBlackWhite(img, color)
                img = self.__flip_img(img, flip)
                if img is not None:
                    shared_img = self.__get_shared_np_array(img)
                    np.copyto(shared_img, img)
                time.sleep(delay)
            if self.pipe.poll():
                msg = self.pipe.recv()
                if msg == "stop":
                    cap.release()
                    break
                elif msg == "pause":
                    while msg != "play":
                        msg = self.pipe.recv()
                elif msg == "gamma":
                    gamma = self.pipe.recv()
                elif msg == "color":
                    color = self.pipe.recv()
                elif msg == "flip":
                    flip = self.pipe.recv()

        cap.release()
        self.capturing.value = 0


    def run(self):
        self.capturing.value = 1
        dev_list = uvc.device_list()
        cap = uvc.Capture(dev_list[self.source]['uid'])
        self.__setup_eye_cam(cap)
        cap.frame_mode = self.mode
        attempt, attempts = 0, 6
        gamma, color, flip = 1, True, False
        while attempt < attempts:     
            try:
                frame = cap.get_frame(2.0)
                img   = self.__adjust_gamma(frame.bgr, gamma)
                img   = self.__cvtBlackWhite(img, color)   
                img   = self.__flip_img(img, flip)       
                if img is not None:
                    attempt = 0
                    shared_img = self.__get_shared_np_array(img)
                    np.copyto(shared_img, img)
            except Exception as e:
                traceback.print_exc()
                cap = self.__reset_mode(cap)
                attempt += 1           
            if self.pipe.poll():
                msg = self.pipe.recv()
                if msg == "stop": 
                    break
                elif msg == "gamma":
                    gamma = self.pipe.recv()
                elif msg == "color":
                    color = self.pipe.recv()
                elif msg == "flip":
                    flip = self.pipe.recv()
        self.capturing.value = 0
        print("camera", self.source, "closed")

    def reset_center_axis(self):
        return None
    
