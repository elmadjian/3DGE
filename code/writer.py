import cv2
import datetime as dt
from multiprocessing import Process
import time
import ctypes
    

class VideoWriter(Process):

    def __init__(self, name, mode, array, pipe):
        Process.__init__(self)
        self.name = name
        self.shared_array = array
        self.mode = mode
        self.pipe = pipe
    
    def _get_recorder(self):
        stamp = dt.datetime.today().strftime('%Y%m%d-%H.%M.%S')
        cam_log_path = 'logs/{}_{}.avi'.format(self.name, stamp)
        #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        resolution = (self.mode[0], self.mode[1])
        fps = self.mode[2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        return cv2.VideoWriter(cam_log_path, fourcc, fps, resolution)

    def _get_shared_np_array(self):
        nparray = np.frombuffer(self.shared_array, dtype=ctypes.c_uint8)
        w, h = self.mode[0], self.mode[1]
        if len(nparray) == h * w * 3:
            return nparray.reshape((h,w,3))
        return np.ones((h,w,3), dtype=ctypes.c_uint8)
    
    def run(self):
        writer = self._get_recorder()
        record = True
        while record:
            time.sleep(1/self.mode[2])
            try:
                img = self._get_shared_np_array()
                img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                writer.write(img_bw)
            except Exception as e:
                pass
            if self.pipe.poll():
                msg = self.pipe.recv()
                if msg == "record":
                    record = self.pipe.recv()
                if msg == "change_mode"
                    self.mode = self.pipe.recv()
                    self.shared_array = self.pipe.recv()

            
