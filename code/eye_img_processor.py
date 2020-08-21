import cv2
import numpy as np
import img_processor as imp
import time
import sys
import ellipse as ell
import eyefitter as ef
import traceback

class EyeImageProcessor(imp.ImageProcessor):

    def __init__(self, source, mode, pipe, array, pos, cap):
        super().__init__(source, mode, pipe, array, pos, cap)
        self.eye_cam = True
        self.bbox = None
        self.tracking = False
        self.lost_tracking = 0
        self.buffer = []
        
        self.intensity_range = 23
        self.bbox_size = {'min': None, 'max': None}
        self.pupil_size = {'min': None, 'max': None}

        #ROI grid config
        self.grid_v = 9
        self.grid_weight = np.ones((self.grid_v, self.grid_v))
        self.grid_id = -1
        
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
                    bbox = self.__get_bbox(c, size, img)
                    if self.__is_consistent(axes, width, 0.050) and bbox is not None:
                        self.bbox = bbox
                        self.grid_weight[self.grid_id] = 1
                        self.__draw_tracking_info(c, img, self.bbox)
                        if mode_3D:
                            self.fitter.unproject_ellipse([c,axes,rad],img)
                            self.fitter.draw_vectors([c,axes,rad], img)
                            ppos = self.fitter.curr_state['gaze_pos'].flatten()
                            return img, np.hstack((ppos,time.monotonic()))
                        return img, np.array([c[0]/width, c[1]/height, time.monotonic(),0])
                    else:
                        self.__perform_penalty(pupil=True)
                else:
                    self.__perform_penalty()
            except Exception:
                traceback.print_exc()
        return img, None

    
    def reset_center_axis(self):
        self.fitter.reset_axis()
        print('>>> resetting center axis')


    def __find_ROI(self, frame):
        '''
        Sliding window that finds an initial good seed for the pupil based
        on image integrals (pupil is usually dark)

        When pupil tracking is failing from the quadrant seed 
        '''
        h = int(frame.shape[0]/3)
        w = int(frame.shape[1]/3)
        hfs = int(h/4)
        wfs = int(w/4)
        self.bbox_size['min'], self.bbox_size['max'] = w //2, w
        self.pupil_size['min'], self.pupil_size['max'] = w//5, w//1.5
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        minval = sys.maxsize
        bbox = None
        for y in range(self.grid_v):
            for x in range(self.grid_v):
                crop = gray[y*hfs:y*hfs+h, x*wfs:x*wfs+w]
                integral = cv2.integral(crop)
                val = integral[-1,-1] * self.grid_weight[y,x]
                if val < minval:
                    minval = val
                    bbox = (x*wfs, y*hfs, w, h)
                    self.grid_id = (y,x)
                #cv2.rectangle(frame, bbox, (255,120,120), 2, 1)
                #cv2.imshow('gente...', frame)
                #cv2.waitKey(0)
        cv2.rectangle(frame, bbox, (255,120,120), 2, 1)
        print('curr_id:', self.grid_id, 'curr_weight:', self.grid_weight[self.grid_id])
        return bbox


    def __perform_penalty(self, pupil=False):
        if not pupil:
            self.buffer = []
        self.lost_tracking += 1
        if self.lost_tracking > 15:
            self.grid_weight[self.grid_id] += 2
            self.tracking = False
            self.lost_tracking = 0


    def __is_consistent(self, axes, width, thresh):
        '''
        The higher the threshold the higher the number of
        potential false positives. Ideally we want only reliable pupil
        candidates.
        '''
        if np.max(axes) > self.pupil_size['max']:
            print('pupil size too big')
            return False
        if np.max(axes) < self.pupil_size['min']:
            print('pupil size too small')
            return False
        axes_np = np.sort(np.array(axes)/width)
        if len(self.buffer) < 4:
            self.buffer.append(axes_np)
        else:
            dist = 0
            for ax in self.buffer:
                dist += np.linalg.norm(ax - axes_np)
            # if dist > thresh:
            #     print('pupil very different from history')
            #     return False
            self.buffer.pop(0)
            self.buffer.append(axes_np)
        return True


    def __draw_tracking_info(self, p, img, bbox, color=(255,120,120)):
        cv2.drawMarker(img, (int(p[0]), int(p[1])), (0,255,0),\
                    cv2.MARKER_CROSS, 12, 1)
        cv2.rectangle(img, bbox, color, 2, 1)


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
        if w < self.bbox_size['min'] or w > self.bbox_size['max']:
            return 
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


    def __fit_ellipse(self, crop, cnt):
        empty_box = np.zeros(crop.shape)
        cv2.drawContours(empty_box, cnt, -1, 255, 2)
        cv2.imshow('fitellipse', empty_box)
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
        edges = self.__find_edges(filtered, 40)
        cv2.imshow('canny', edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        edges  = cv2.dilate(edges, kernel)
        edges  = self.__filter_edges(edges)
        cv2.imshow('filtered', edges)
        cnt,_ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        #cnts = self.__test_curvature(cnt)
        cnts  = [(cv2.contourArea(c), c) for c in cnt]
        if cnts:
            biggest = max(cnts, key=lambda x: x[0])[1]
            #print('biggest:', biggest)
            hull = cv2.convexHull(biggest)
            if len(hull) >= 5:
                ellipse = self.__fit_ellipse(cropgray, hull)
                painted = cv2.cvtColor(cropgray, cv2.COLOR_GRAY2BGR)
                frame[y+3:y+h-3, x+3:x+w-3] = painted
                if ellipse is not None:
                    return ellipse


    def __test_curvature(self, contours, cutoff_min=90, cutoff_max=165):
        '''
        cutoff_min: min curvature in degrees between 3 points
        cutoff_max: max curvature in degrees between 3 points
        '''
        new_contours = []
        for c in contours:
            c = c[:len(c)//2]
            if len(c) <= 5:
                continue 
            approx_curve = cv2.approxPolyDP(c, 1.5, False)
            for i in range(1, len(approx_curve)-1):
                A = approx_curve[i-1]
                B = approx_curve[i]
                C = approx_curve[i+1]
                BA = A[0]-B[0]
                BC = C[0]-B[0]
                cos_angle = np.dot(BA,BC)/(np.linalg.norm(BA)*np.linalg.norm(BC))
                angle = np.degrees(np.arccos(cos_angle))
                print('angle:', angle)
                if angle > cutoff_min and angle < cutoff_max:
                    new_contours.append(A)
                    new_contours.append(B)
                    new_contours.append(C)
            print('_____')
        print('-------------')
        new_contours = np.array(new_contours).reshape((-1,1,2)).astype(np.int32)
        #print(new_contours)
        return new_contours

    # def __test_distance(self, contours, img):
    #     c = self.__blob_center(img)
    #     cv2.circle(img, c, 7, (255,0,255), img, -1)
    #     cnt_centers = set()
    #     for cnt in contours:
    #         p = len(cnt)//2
    #         cnt_centers.append(cnt[p][0])
    #     for cnt in cnt_centers:
    #         dist_c = np.linalg.norm(cnt-c)
    #         for p in cnt_centers:
    #             if cnt == p:
    #                 continue
                


    def __filter_edges(self, edges):
        '''
        cuttof:
        ratio:
        '''
        cutoff = (edges.shape[0] + edges.shape[1])/20
        _,labels, stats,_ = cv2.connectedComponentsWithStats(edges)
        filtered = np.zeros(edges.shape, np.uint8)
        stats = stats[1:]
        idx = 0
        if len(stats) > 0:
            for i in range(len(stats)):
                ratio = stats[i,2]/stats[i,3]
                if (0.25 < ratio < 2) and stats[i,4] > cutoff:
                    idx = i+1
                    filtered[labels==idx] = 255
        return filtered


    def __blob_center(self, img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('testando', thresh)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        x = stats[1,0]
        y = stats[1,1]
        w = stats[1,2]
        h = stats[1,3]
        return (x+w//2, y+h//2)


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


    def __find_edges(self, img, intensity_values):
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
        spec_mask = cv2.inRange(img, np.array(0), np.array(highest_spike_index-5))
        spec_mask = cv2.erode(spec_mask, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        edges = cv2.Canny(img, 160, 160*2, apertureSize=5)
        edges = cv2.min(edges, spec_mask)
        edges = cv2.min(edges, bin_img)
        return edges