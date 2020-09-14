import cv2
import numpy as np
import img_processor as imp
import time
import sys
import ellipse as ell
import eyefitter as ef
import traceback
import skimage as si

from pupil_detectors import Detector3D

import matplotlib.pyplot as plt

class EyeImageProcessor(imp.ImageProcessor):

    def __init__(self, source, mode, pipe, array, pos, cap):
        super().__init__(source, mode, pipe, array, pos, cap)
        self.eye_cam = True
        self.bbox = None
        self.detector = Detector3D()
        self.tracking = False
        self.lost_tracking = 0
        self.buffer = {'a':[], 'b':[]}
        
        self.intensity_range = 23
        self.bbox_size = {'min': None, 'max': None}
        self.pupil_size = {'min': None, 'max': None}

        #ROI grid config
        self.grid_v = 9
        self.center = None
        self.consistency = False
        
        #3D
        sensor_size = (3.6, 4.8)
        focal_length = 6
        res = (mode[1], mode[0])
        self.fitter = ef.EyeFitter(focal_length, res, sensor_size)

        # print(self.detector.get_properties())
        # print('-----')


    def process(self, img, mode_3D=False):
        if img is None:
            return None, None
        height, width = img.shape[0], img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # result = self.detector.detect(gray, 0.5)
        # if result["confidence"] > 0.5:
        #     ellipse = result["ellipse"]
        #     c = ellipse["center"]
        #     self.__draw_tracking_info(ellipse, img)
        #     return img, np.array([c[0]/width, c[1]/height, time.monotonic(),0])
            


        #height, width = img.shape[0], img.shape[1]   
        # if not self.tracking:
        #     self.bbox = self.__find_ROI(img)
        #     self.tracking = True
        # else:
        #     x,y,_,_ = self.bbox
        #     try:
        #         pupil = self.__find_contours(self.bbox, img)
        #         if pupil is not None:
        #             c, axes, rad = pupil.get_parameters()
        #             c = (c[0]+x+3, c[1]+y+3)
        #             size = max(axes)*2
        #             bbox = self.__get_bbox(c, size, img)
        #             if self.__is_consistent(pupil, 10) and bbox is not None:
        #                 self.consistency = True
        #                 self.bbox = bbox
        #                 self.__draw_tracking_info([c,axes,rad], img, self.bbox)
        #                 if mode_3D and len(self.buffer['a']) >= 3:
        #                     self.fitter.unproject_ellipse([c,axes,rad],img)
        #                     self.fitter.draw_vectors([c,axes,rad], img)
        #                     ppos = self.fitter.curr_state['gaze_pos'].flatten()
        #                     return img, np.hstack((ppos,time.monotonic()))
        #                 return img, np.array([c[0]/width, c[1]/height, time.monotonic(),0])
        #             # else:
        #             #     self.__perform_penalty(pupil=True)
        #         else:
        #             self.__perform_penalty()
        #     except Exception:
        #         traceback.print_exc()
        return img, None


    def reset_center_axis(self):
        self.fitter.reset_axis()
        print('>>> resetting center axis')

    
    def __get_weight(self, mpoint):
        distance = np.linalg.norm(self.center-mpoint)
        weight = 1
        if distance < 0.6*self.center[1]:
            weight -= 0.1
        if distance < 0.4*self.center[1]:
            weight -= 0.25
        if distance < 0.25*self.center[1]:
            weight -= 0.35
        return weight


    def __find_ROI(self, frame):
        '''
        Sliding window that finds an initial good seed for the pupil based
        on image integrals (pupil is usually dark)

        If no pupil center has been successfully found before, then 
        it assumes that the pupil is probably in the center region of the image
        '''
        h = int(frame.shape[0]/3)
        w = int(frame.shape[1]/3)
        hfs = int(h/4)
        wfs = int(w/4)
        if self.center is None:
            self.center = np.array([frame.shape[0]//2, frame.shape[1]//2])
        self.bbox_size['min'], self.bbox_size['max'] = w //3, w*1.3
        self.pupil_size['min'], self.pupil_size['max'] = w//6, w//2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        minval = sys.maxsize
        bbox = None
        for y in range(self.grid_v):
            for x in range(self.grid_v):
                crop = gray[y*hfs:y*hfs+h, x*wfs:x*wfs+w]
                mpoint = np.array([y*hfs+h//2, x*wfs+w//2])
                integral = cv2.integral(crop)
                weight = self.__get_weight(mpoint)
                val = integral[-1,-1] * weight
                if val < minval:
                    minval = val
                    bbox = (x*wfs, y*hfs, w, h)
        #cv2.rectangle(frame, bbox, (255,120,120), 2, 1)
        return bbox


    def __perform_penalty(self):
        self.lost_tracking += 1
        if self.lost_tracking > 15:
            #print('tracking lost')
            self.tracking = False
            self.lost_tracking = 0
            self.buffer = {'a':[], 'b':[]}
            if not self.consistency:
                self.center = None
            self.consistency = False


    def __is_consistent(self, ellipse, thresh):
        '''
        The higher the threshold the higher the number of
        potential false positives. Ideally we want only reliable pupil
        candidates.
        '''
        center, axes, _ = ellipse.get_parameters()
        if np.max(axes) > self.pupil_size['max'] or\
        np.max(axes) < self.pupil_size['min']:
            return False
        a, b = np.max(axes), np.min(axes)
        if len(self.buffer['a']) < 3:
            self.buffer['a'].append(a)
            self.buffer['b'].append(b)
        else:
            a_var = np.var(self.buffer['a'])
            b_var = np.var(self.buffer['b'])
            if a_var + b_var > thresh:
                #print("not consistent")
                self.buffer = {'a':[], 'b':[]}
                return False
            self.buffer['a'].pop(0)
            self.buffer['b'].pop(0)
            self.buffer['a'].append(a)
            self.buffer['b'].append(b)
            self.center = center
        return True


    def __draw_tracking_info(self, ellipse, img, color=(255,120,120)):
        center = tuple(int(v) for v in ellipse["center"])
        axes = tuple(int(v/2) for v in ellipse["axes"])
        rad = ellipse["angle"]
        cv2.drawMarker(img, center, (0,255,0), cv2.MARKER_CROSS, 12, 1)
        #cv2.rectangle(img, bbox, color, 2, 1)
        cv2.ellipse(img, center, axes, rad, 0, 360, (0,0,255), 2)


    def __get_bbox(self, point, size, img):
        x1 = point[0]-size*0.7
        y1 = point[1]-size*0.7
        x2 = point[0]+size*0.7
        y2 = point[1]+size*0.7
        x1 = self.__test_boundaries(x1, img.shape[1])
        y1 = self.__test_boundaries(y1, img.shape[0])
        x2 = self.__test_boundaries(x2, img.shape[1])
        y2 = self.__test_boundaries(y2, img.shape[0])
        w = x2-x1
        h = y2-y1
        if w < self.bbox_size['min'] or w > self.bbox_size['max']:
            return 
        return int(x1),int(y1),int(w),int(h)

    def __test_boundaries(self, x, lim):
        if x < 0:
            return 0
        if x >= lim:
            return lim-1
        return x


    def __fit_ellipse(self, crop, cnt):
        empty_box = np.zeros(crop.shape)
        cv2.drawContours(empty_box, cnt, -1, 255, 2)
        points = np.where(empty_box == 255)
        vertices = np.array([points[0], points[1]]).T
        ellipse = ell.Ellipse([vertices[:,1], vertices[:,0]])
        return ellipse
        

    def __find_contours(self, bbox, frame):
        x,y,w,h = bbox
        crop = frame[y+3:y+h-3, x+3:x+w-3]
        cropgray = crop
        if len(cropgray.shape) > 2:
            cropgray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(cropgray, 7, 20, 20)
        edges = self.__find_edges(filtered, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        edges  = self.__filter_edges(edges)
        cnt, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        pts = self.__get_curve_points(cnt, edges)
        if len(pts) >= 5:
            hull = cv2.convexHull(pts)
            if len(hull) >= 5:
                ellipse = self.__fit_ellipse(cropgray, hull)
                painted = cv2.cvtColor(cropgray, cv2.COLOR_GRAY2BGR)
                frame[y+3:y+h-3, x+3:x+w-3] = painted
                return ellipse


    def __get_curve_points(self, contours, img, cutoff=72):
        '''
        '''
        new_contours = []
        #contours = sorted(contours, key=lambda x: len(x))
        m = cv2.moments(img)
        centroid = np.array([0,0])
        if m['m00'] != 0:
            centroid = np.array([m['m01']/m['m00'], m['m10']/m['m00']])
        for c in contours:
            approx_curve = cv2.approxPolyDP(c, 1.5, False)
            length = len(approx_curve)
            p = approx_curve[length//2][0]
            dist = np.linalg.norm(centroid-p)
            if dist > cutoff:
                continue
            for i in range(len(approx_curve)):
                new_contours.append(approx_curve[i][0])
                mytuple = (approx_curve[i][0][0], approx_curve[i][0][1])
        new_contours = np.array(new_contours).reshape((-1,1,2)).astype(np.int32)
        return new_contours


    def __filter_edges(self, edges):
        '''
        min_area: connected component minimum area
        ratio: we want components with a certain curvature
        returns a zeroed image if there are too many edges (i.e., eyelashes)
        or too few
        '''
        min_area = (edges.shape[0] + edges.shape[1])/18
        _, labels, stats, _ = cv2.connectedComponentsWithStats(edges)
        filtered = np.zeros(edges.shape, np.uint8)
        stats = stats[1:]
        idx, val_area = 0, 0
        if len(stats) > 0:
            for i in range(len(stats)):
                ratio = stats[i,2]/stats[i,3]
                if (0.15 < ratio < 6) and stats[i,4] > min_area:
                    idx = i+1
                    filtered[labels == idx] = 255
                val_area += stats[i,4]
        if val_area/min_area > 29 or val_area/min_area < 10:
            filtered = np.zeros(edges.shape, np.uint8)
        return filtered


    def __find_edges(self, img, intensity_values=25):
        '''
        Adapted edge detector from Pupil Labs
        '''
        hist = np.bincount(img.ravel(), minlength=256)
        lowest_spike_index = 255
        highest_spike_index = 0
        max_intensity = 0
        found_section = False
        for i in range(len(hist)):
            intensity = hist[i]
            if intensity > intensity_values:
                max_intensity = np.maximum(intensity, max_intensity)
                lowest_spike_index = np.minimum(lowest_spike_index, i)
                highest_spike_index = np.maximum(highest_spike_index, i)
                found_section = True
        if not found_section:
            lowest_spike_index = 200
            highest_spike_index = 255
        bin_img = cv2.inRange(img, np.array(0), np.array(lowest_spike_index + self.intensity_range))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        bin_img = cv2.dilate(bin_img, kernel, iterations=2)
        spec_mask = cv2.inRange(img, np.array(0), np.array(highest_spike_index - 5))
        spec_mask = cv2.erode(spec_mask, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(img, 160, 160*2, apertureSize=5)
        edges = cv2.min(edges, spec_mask)
        edges = cv2.min(edges, bin_img)
        return edges