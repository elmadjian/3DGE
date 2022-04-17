import subprocess
import re
import uvc
from PySide2.QtCore import QObject, Signal, Slot, Property

class VideoIO_UVC(QObject):

    def __init__(self):
        QObject.__init__(self)
        self.cameras = {}
        self.read_inputs()
        self.leye  = None
        self.reye  = None

    
    def read_inputs(self):
        dev_list = uvc.device_list()
        for i in range(len(dev_list)):
            name = dev_list[i]['name']
            self.cameras[i] = name


    @Property('QVariantList')
    def camera_list(self):
        self.read_inputs()
        cameras = ["{}: {}".format(i,self.cameras[i]) for i in self.cameras.keys()]
        opts = ['No feed', 'File...']
        return opts + cameras


    def get_camera_name(self, source):
        return self.cameras[source]

    
    def set_active_cameras(self, leye, reye):
        self.leye  = leye
        self.reye  = reye

    @Slot(bool)
    def stop_leye_cam(self, video_file):
        self.leye.stop(video_file)

    @Slot(bool)
    def stop_reye_cam(self, video_file):
        self.reye.stop(video_file)

    @Slot(bool)
    def stop_cameras(self, video_file):
        print(">>> Closing video feed...")
        self.leye.stop(video_file)
        self.reye.stop(video_file)
        print(">>> Finished!")

    @Slot(bool, bool, bool)
    def play_cams(self, leye_t, reye_t):
        self.leye.play(leye_t)
        self.reye.play(reye_t)

    @Slot(bool, bool, bool)
    def pause_cams(self, leye_t, reye_t):
        self.leye.pause(leye_t)
        self.reye.pause(reye_t)

    @Slot(str, str)
    def load_video(self, cam_id, filename):
        if cam_id.startswith("Left"):
            self.leye.stop(video_file=True)
            self.leye.set_video_file(filename)
        else:
            self.reye.stop(video_file=True)
            self.reye.set_video_file(filename)

    # @Slot(str)
    # def set_camera_last_session(self, cam_id):
    #     if cam_id.startswith('Left'):
    #         self.leye.load_last_session_cam()
    #     else:
    #         self.reye.load_last_session_cam()

    @Slot(str, str)
    def set_camera_source(self, cam_id, cam_name):
        source = int(cam_name.split(':')[0])
        print('SOURCE:', source)
        if cam_id.startswith("Left"):
            self.__change_cameras(self.leye, self.reye, source)
        else:
            self.__change_cameras(self.reye, self.leye, source)

    @Slot()
    def save_session_config(self):
        print('>>> Saving session configuration...')
        

    def __change_cameras(self, cam1, cam2, source):
        '''
        cam1 is the camera to have its source switched
        '''
        cam1.stop()
        prev_source = cam1.get_source()
        if source == cam2.get_source():
            cam2.stop()
            cam2.set_source(prev_source)
        cam1.set_source(source)



if __name__=="__main__":
    v = VideoIO_UVC()
    print(v.get_cameras())