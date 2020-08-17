import cv2
import numpy as np
import img_processor as imp
import time
import sys
import ellipse as ell
import eyefitter as ef

class EyeImageProcessor(imp.ImageProcessor):

    def __init__(self, source, mode, pipe, array, pos, cap):
        super().__init__(source, mode, pipe, array, pos, cap)
        self.eye_cam = True
        self.bbox = None
        self.tracking = False
        self.lost_tracking = 0
        self.buffer = []
        
        #3D
        sensor_size = (3.6, 4.8)
        focal_length = 6
        res = (mode[1], mode[0])
        self.fitter = ef.EyeFitter(focal_length, res, sensor_size)
        

    def process(self, img, mode_3D=False):
        if img is None:
            return None, None
        height, width = img.shape[0], img.shape[1]
        if not self.tracking:
            self.bbox = self.__find_ROI(img)
            self.tracking = True
        else:
            x,y,_,_ = self.bbox
            try:
                pupil = self.__find_contours(self.bbox, img)
                if pupil is not None:
                    c, axes, rad = pupil
                    c = (c[0]+x+3, c[1]+y+3)
                    size = max(axes)*2
                    self.bbox = self.__get_bbox(c, size, img)
                    if self.__is_consistent(axes, width, 0.050):
                        self.__draw_tracking_info(c, img)
                        if mode_3D:
                            self.fitter.unproject_ellipse([c,axes,rad],img)
                            self.fitter.draw_vectors([c,axes,rad], img)
                            ppos = self.fitter.curr_state['gaze_pos'].flatten()
                            return img, np.hstack((ppos,time.monotonic()))
                        return img, np.array([c[0]/width, c[1]/height,
                                time.monotonic(),0], dtype='float32')
                else:
                    self.buffer = []
                    self.lost_tracking += 1
                    if self.lost_tracking > 20:
                        self.tracking = False
                        self.lost_tracking = 0
            except Exception as e:
                print('Error:', e)
        return img, None

    
    def reset_center_axis(self):
        self.fitter.reset_axis()
        print('>>> resetting center axis')
        


    def __is_consistent(self, axes, width, thresh):
        '''
        The higher the threshold the higher the number of
        potential false positives
        '''
        axes_np = np.sort(np.array(axes)/width)
        if len(self.buffer) < 3:
            self.buffer.append(axes_np)
        else:
            dist = 0
            for ax in self.buffer:
                dist += np.linalg.norm(ax - axes_np)
            if dist > thresh:
                self.buffer.pop(0)
                return False
            self.buffer.pop(0)
            self.buffer.append(axes_np)
        return True


    def __draw_tracking_info(self, p, img):
        cv2.drawMarker(img, (int(p[0]), int(p[1])), (0,255,0),\
                    cv2.MARKER_CROSS, 12, 1)
        cv2.rectangle(img, self.bbox, (255,120,120), 2, 1)


    def __get_bbox(self, point, size, img):
        x1 = point[0]-size*0.8
        y1 = point[1]-size*0.8
        x2 = point[0]+size*0.8
        y2 = point[1]+size*0.8
        x1 = self.__test_boundaries(x1, img.shape[1])
        y1 = self.__test_boundaries(y1, img.shape[0])
        x2 = self.__test_boundaries(x2, img.shape[1])
        y2 = self.__test_boundaries(y2, img.shape[0])
        w = x2-x1
        h = y2-y1
        return int(x1),int(y1),int(w),int(h)

    def __test_boundaries(self, x, lim):
        if x < 0:
            return 0
        if x >= lim:
            return lim-1
        return x

    def __remove_glint(self, img, edges):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
        dilate = cv2.dilate(thresh, kernel)
        edges[dilate >= 255] = 0
        return edges

    # def __find_glint(self, img):
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     thresh_3 = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)[1]
    #     cv2.imshow('thresh_3', thresh_3)        


    def __fit_ellipse(self, crop, cnt):
        empty_box = np.zeros(crop.shape)
        cv2.drawContours(empty_box, cnt, -1, 255, 2)
        points = np.where(empty_box == 255)
        vertices = np.array([points[0], points[1]]).T
        ellipse = ell.Ellipse([vertices[:,1], vertices[:,0]])
        return ellipse.get_parameters()
        

    def __find_contours(self, bbox, frame):
        x,y,w,h = bbox
        crop = frame[y+3:y+h-3, x+3:x+w-3]
        cropgray = crop
        if len(cropgray.shape) > 2:
            cropgray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(cropgray, 7, 20, 20)
        adjusted = self.__adjust_histogram(filtered)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        upper, thresh = cv2.threshold(adjusted,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        lower = 0.5*upper
        edges  = cv2.Canny(adjusted, lower, upper)
        edges  = self.__remove_glint(filtered, edges)
        edges  = self.__filter_edges(edges)
        dilate = cv2.dilate(edges, kernel, iterations=2)
        cnt,_ = cv2.findContours(dilate, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cnts  = [(cv2.contourArea(c), c) for c in cnt]
        if cnts:
            biggest = max(cnts, key=lambda x: x[0])[1]
            hull = cv2.convexHull(biggest)
            if len(hull) >= 5:
                ellipse = self.__fit_ellipse(cropgray, hull)
                painted = cv2.cvtColor(cropgray, cv2.COLOR_GRAY2BGR)
                frame[y+3:y+h-3, x+3:x+w-3] = painted
                if ellipse is not None:
                    return ellipse


    def __find_ROI(self, frame):
        h = int(frame.shape[0]/3)
        w = int(frame.shape[1]/3)
        hfs = int(h/6)
        wfs = int(w/6)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        minval = sys.maxsize
        bbox = None
        for y in range(13):
            for x in range(13):
                crop = gray[y*hfs:y*hfs+h, x*wfs:x*wfs+w]
                integral = cv2.integral(crop)
                copy = frame.copy()
                if integral[-1,-1] < minval:
                    minval = integral[-1,-1]
                    bbox = (x*wfs, y*hfs, w, h)
        cv2.rectangle(frame, bbox, (255,120,120), 2, 1)
        return bbox

    
    def __filter_edges(self, edges):
        cutoff = (edges.shape[0] + edges.shape[1])/8
        _,labels, stats,_ = cv2.connectedComponentsWithStats(edges)
        filtered = np.zeros(edges.shape, np.uint8)
        stats = stats[1:]
        length, idx, = 0, 0
        if len(stats) > 0:
            for i in range(len(stats)):
                ratio = stats[i,2]/stats[i,3]
                if (0.33 < ratio < 3) and stats[i,4] > cutoff:
                    idx = i+1
                    filtered[labels==idx] = 255
        return filtered


    def __adjust_histogram(self, img):
        hist = np.bincount(img.ravel(),minlength=256)
        win  = 16
        hmax, hmin, cutoff = 100, sys.maxsize, 0
        for i in range(0,256-win,2):
            sample = hist[i:i+win]
            ssum   = np.sum(sample)
            if ssum > hmax:
                hmax = ssum
            elif hmax > 100 and ssum < hmin:
                hmin = ssum
            if hmax > 100 and ssum > hmin:
                cutoff = i + win/2
                break
        lower, upper = img.copy(), img.copy()
        lower, upper = lower.astype('float32'), upper.astype('float32')
        lower[lower > cutoff] = 30
        upper[upper <= cutoff] = -30
        lower -= 30
        upper += 30
        lower = np.clip(lower, 0, 255)
        upper = np.clip(upper, 0, 255)
        lower, upper = lower.astype('uint8'), upper.astype('uint8')
        merged = cv2.add(lower, upper)
        return merged