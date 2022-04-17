import cv2
import numpy as np
import time
import os
import re
import socket
import traceback
import data_storage as ds
from PySide2.QtCore import QObject, Signal, Slot, Property
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from threading import Thread
import yaml


class HMDCalibrator(QObject):

    move_on = Signal()
    conn_status = Signal(bool)

    def __init__(self, v_targets, h_targets, samples_per_tgt, timeout):
        '''
        ntargets: number of targets that are going to be shown for calibration
        frequency: value of the tracker's frequency in Hz
        '''
        QObject.__init__(self)
        self.target_list = self._generate_target_list(v_targets, h_targets)
        self.storer = ds.Storer()
        self.left_mat, self.right_mat = None, None
        self.current_target = -1
        self.leye, self.reye = None, None
        self.samples = samples_per_tgt
        self.timeout = timeout
        self.collector = None
        self.predictor = None
        self.stream = False
        self.storage = False
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", 50021))
        self.ip, self.port = self.load_network_options()


    def set_sources(self, leye, reye):
        self.leye = leye
        self.reye = reye
        self.storer.set_sources(leye, reye)

    def set_vergence_control(self, vergence):
        self.vergence = vergence

    def _freeze_models(self):
        self.leye.freeze_model()
        self.reye.freeze_model()

    def _unfreeze_models(self):
        self.leye.unfreeze_model()
        self.reye.unfreeze_model()

    def load_network_options(self):
        ip, port = "", ""
        filename = os.path.join('config', 'hmd_config.yaml')
        with open(filename, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            ip = data['ip'] 
            port = data['port']
        return ip, port    

    def _generate_target_list(self, v, h):
        '''
        Generate a uniformly distributed set of calibration targets and test 
        targets based on a predefined number of vertical and horizontal 
        targets on object construction.
        '''
        target_list = []
        for y in np.linspace(-1,1, v):
            for x in np.linspace(-1,1, h):
                target_list.append(np.array([x,y,1], dtype=np.float32))
        #seed = np.random.randint(0,99)
        #rnd  = np.random.RandomState(seed)
        #rnd.shuffle(target_list)
        return target_list

       
    @Property('QVariantList')
    def target(self):
        if self.current_target >= len(self.target_list):
            return [-9,-9]
        tgt = self.target_list[self.current_target]
        return [float(tgt[0]), float(tgt[1])]


    @Slot()
    def store_calibration(self):
        self.storer.store_calib_debug()
        print('>>> calibration saved')
     
    @Property(str)
    def hmd_ip(self):
        return self.ip

    @Property(int)
    def hmd_port(self):
        return self.port        

    @Slot(str, str)
    def update_network(self, ip, port):
        self.ip, self.port = ip, int(port)
        filename = os.path.join('config', 'hmd_config.yaml')
        data = {}
        data['ip'] = ip
        data['port'] = self.port
        with open(filename, 'w') as f:
            yaml.dump(data, f)

    @Slot()
    def connect(self):
        '''
        1. Called when the user wants to connect to the HMD.
        Upon successful connection, calibration routine starts

        worfklow: -> [QML] 'connect' button -> [Python] self.connect() 
                  -> [Unity] UDP('C') -> [Unity] GazeManager.StartCalibration() 
                  -> [Unity] Load calibration scene -> [Python] UDP('C')
                  -> [QML] connStatus update -> [Python] self.start_calibration()
        '''
        self.socket.settimeout(10)
        try:
            self.socket.sendto('C'.encode(), (self.ip, self.port))
            response = self.socket.recv(1024).decode()
            if response:
                self.conn_status.emit(True)
                self.start_calibration()
        except Exception as e:
            self.conn_status.emit(False)
            print("Connection error:", e)

    
    def start_calibration(self):
        '''
        2. Called upon successfull connection. It unfreezes any previous
        saved model, initiliazes storage, and clear other variables
        used in calibration

        workflow: -> [QML] 'space' or 'double click' starts calibration
                  -> [QML] checkStateAndNextStep() -> [QML] nextCalibStep()
        '''
        print('resetting calibration')
        self._unfreeze_models()
        self.storer.initialize_storage(len(self.target_list))
        self.left_mat = None
        self.right_mat = None
        if self.predictor is not None:
            self.stream = False
            self.predictor.join()
        self.current_target = -1


    @Slot()
    def next_target(self):
        '''
        3. Called every time we move to a new calibration target
        using 'space' or 'double click' (nextCalibStep() in QML)

        workflow: -> [Python] self.next_target() -> [Python] get next target
                  -> [Python] send target coordinates
                  -> [Unity] UDP('N') -> [Unity] GazeManager.NextTarget()
                  -> [QML] target value updated

                  if there is no next target -> [Unity] UDP('F')
                  -> [Unity] GazeManager.StopCalibration() -> [Python] UDP('F')
                  -> [Unity] load experiment -> [QML] call gaze estimation
                  -> [Python] self.perform_estimation()
        '''
        if self.collector is not None:
            self.collector.join()
        self.current_target += 1
        if self.current_target >= len(self.target_list):
            self.socket.sendto("F".encode(), (self.ip, self.port))
            return
        tgt = self.target_list[self.current_target]
        msg = 'N:' + str(tgt[0]) + ':' + str(tgt[1]) + ':' + str(tgt[2])
        self.socket.sendto(msg.encode(), (self.ip, self.port))


    @Slot(int, int)
    def collect_data(self, minfq, maxfq):
        '''
        4. This is called every time we are recording eye data.
        It alerts the HMD to show the corresponding visual feedback

        workflow: -> [QML] get eye cam refresh rates -> [Python] self.collect_data()
                  -> [Python] self._get_target_data (threaded)
        '''
        msg = 'R'.encode()
        self.socket.sendto(msg, (self.ip, self.port))
        vecs = self.socket.recv(1024).decode()
        self.collector = Thread(target=self._get_target_data, args=(vecs,minfq,maxfq,))
        self.collector.start()


    def _get_target_data(self, vecs, maxfreq, minfreq):
        '''
        5. Captures data from the eye img processor associated with
        a particular target 'idx'

        workflow: -> [Python] self.storer.collect_data()
                  -> [Python] emit 'move_on' -> [QML] nextCalibStep()
                  -> [Python] back to step 3
        '''
        idx = self.current_target
        t = time.time()
        lct, rct = self.storer.l_centers, self.storer.r_centers
        vecs = re.sub('\(', '', vecs)
        vecs = re.sub('\)', '', vecs)
        _, l_vec, r_vec = vecs.split(';')
        self.storer.l_targets[idx] = np.array([float(v) for v in l_vec.split(',')])
        self.storer.r_targets[idx] = np.array([float(v) for v in r_vec.split(',')])

        while (len(lct[idx]) < self.samples) and (len(rct[idx]) < self.samples)\
        and (time.time()-t < self.timeout):
            self.storer.collect_data(idx, minfreq)
            lct, rct = self.storer.l_centers, self.storer.r_centers
            time.sleep(1/maxfreq)
        self.move_on.emit()
        print("number of samples collected: l->{}, r->{}".format(
            len(self.storer.l_centers[idx]),
            len(self.storer.r_centers[idx])))
        self.storer.replace_by_median(idx)


    @Slot()
    def perform_estimation(self):
        '''
        6. Finds the rotation matrix that transforms eye normals
        from eye camera to HMD coordinate space.
        We use the least squares method.
        
        workflow: -> [Python] np.linalg.lstsq() -> [Python] self._freeze_models()
                  -> [Python] self.predict() (threaded)
        '''       
        if self.leye.is_cam_active(): 
            l_tgt = self.storer.get_l_targets_list()
            l_norm = self.storer.get_l_centers_list()
            row_1 = np.linalg.lstsq(l_norm, l_tgt.T[0,:])[0]
            row_2 = np.linalg.lstsq(l_norm, l_tgt.T[1,:])[0]
            row_3 = np.linalg.lstsq(l_norm, l_tgt.T[2,:])[0]
            self.left_mat = np.vstack((row_1, row_2, row_3))
        if self.reye.is_cam_active():
            r_tgt = self.storer.get_r_targets_list()
            r_norm = self.storer.get_r_centers_list()
            row_1 = np.linalg.lstsq(r_norm, r_tgt.T[0,:])[0]
            row_2 = np.linalg.lstsq(r_norm, r_tgt.T[1,:])[0]
            row_3 = np.linalg.lstsq(r_norm, r_tgt.T[2,:])[0]
            self.right_mat = np.vstack((row_1, row_2, row_3))
        print("Gaze estimation finished")
        if self.storage:
            self.storer.store_calibration()
        self.stream = True
        self._freeze_models()
        self.predictor = Thread(target=self.predict, args=())
        self.predictor.start()


    def predict(self):
        '''
        7. Performs continuous predictions of gaze direction
        based on the calibrated eye models

        worfklow: -> [Unity] send demand -> [Python] UDP('G')
                  -> [Python] self._predict() -> [Unity] UDP('G')
                  -> [Unity] GazeManager.UpdateEyes()
                  -> [Unity] Interactionmanager.SetGazeDirection()
        '''
        count = 0
        while self.stream:
            try:
                demand = self.socket.recv(1024).decode()
                if demand.startswith('G'):
                    data = self._predict()
                    x1, y1, z1 = data[0], data[1], data[2]
                    x2, y2, z2 = data[3], data[4], data[5]
                    x1, y1, z1 = '{:.8f}'.format(x1), '{:.8f}'.format(y1), '{:.8f}'.format(z1)
                    x2, y2, z2 = '{:.8f}'.format(x2), '{:.8f}'.format(y2), '{:.8f}'.format(z2)
                    msg = 'G:'+x1+':'+y1+':'+z1+':'+x2+':'+y2+':'+z2
                    self.socket.sendto(msg.encode(), (self.ip, self.port))
            except Exception as e:
                traceback.print_exc()
                print("no request from HMD...", e)
                count += 1
                if count > 3:
                    self.stream = False
                    break

    def _predict(self):
        pred = [-9,-9,-9,-9,-9,-9]
        if self.left_mat is not None:
            le = self.leye.get_processed_data()
            if le is not None:
                l_transform = np.dot(self.left_mat, le[:3])
                pred[0], pred[1], pred[2] = l_transform
        if self.right_mat is not None:
            re = self.reye.get_processed_data()
            if re is not None:
                r_transform = np.dot(self.right_mat, re[:3])
                pred[3], pred[4], pred[5] = r_transform
        return pred


    @Slot()
    def toggle_storage(self):
        self.storage = not self.storage

    @Slot()
    def save_session(self):
        if self.storage:
            self.storer.store_session()
