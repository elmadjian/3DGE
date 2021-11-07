import numpy as np 
import os
import time

class Storer():
    '''
    Important:
    ---------
    -> 2D: x, y, time, 0, 0, 0, 0
    -> 3D: x_p, y_p, z_p, x_n, y_n, z_n, time
    '''

    def __init__(self):
        self.l_targets, self.r_targets = None, None
        self.l_centers, self.r_centers = None, None
        self.t_imgs, self.l_imgs, self.r_imgs = None, None, None
        self.l_sess, self.r_sess, self.l_raw, self.r_raw = [],[],[],[]
        self.leye, self.reye = None, None
        self.uid = time.ctime().replace(':', '_')
   
    def initialize_storage(self, ntargets):
        self.l_targets = {i:np.empty((0,3), dtype='float32') for i in range(ntargets)}
        self.r_targets = {i:np.empty((0,3), dtype='float32') for i in range(ntargets)}
        self.l_centers = {i:np.empty((0,3), dtype='float32') for i in range(ntargets)}
        self.r_centers = {i:np.empty((0,3), dtype='float32') for i in range(ntargets)}
        self.l_imgs = {i:[] for i in range(ntargets)}
        self.r_imgs = {i:[] for i in range(ntargets)}

   
    def set_sources(self, leye, reye):
        '''
        Set eye cam feed providers
        '''
        self.leye  = leye
        self.reye  = reye    
  
    def collect_data(self, idx, minfreq):
        le = self.leye.get_processed_data()
        re = self.reye.get_processed_data()
        le_img = self.leye.get_np_image()
        re_img = self.reye.get_np_image()
        if self.__check_data_n_timestamp(le, re, 1/minfreq):
            self.__add_data(le, re, idx)
            self.__add_imgs(le_img, re_img, idx)
    
    def __add_data(self, le, re, idx):
        if self.leye.is_cam_active():
            led = np.array([le[0],le[1],le[2]])#,le[3],le[4],le[5]])
            self.l_centers[idx] = np.vstack((self.l_centers[idx], led))
        if self.reye.is_cam_active():
            red = np.array([re[0],re[1],re[2]])#,le[3],le[4],le[5]])
            self.r_centers[idx] = np.vstack((self.r_centers[idx], red))

    def __add_imgs(self, le, re, idx):
        if self.leye.is_cam_active():
            self.l_imgs[idx].append(le)
        if self.reye.is_cam_active():
            self.r_imgs[idx].append(re)

      
    def __check_data_n_timestamp(self, le, re, thresh):
        if le is None and self.leye.is_cam_active():
            return False
        if re is None and self.reye.is_cam_active():
            return False
        le_t, re_t = le[3], re[3]
        if le.any() and re.any():
            if abs(le_t - re_t) < thresh:
                return True
        if le.any() or re.any():
            return True
        return False

    def replace_by_median(self, idx):
        '''
        Sets the median vector of the list as the
        "correct" one in relation to a target
        '''
        l_med = np.median(self.l_centers[idx], axis=0)
        r_med = np.median(self.r_centers[idx], axis=0)
        self.l_centers[idx] = l_med
        self.r_centers[idx] = r_med


    def __dict_to_list(self, dic):
        new_list = np.empty((0,3), dtype='float32')
        for t in dic.keys():
            new_list = np.vstack((new_list, dic[t]))
        return new_list

    def get_l_targets_list(self):
        return self.__dict_to_list(self.l_targets)

    def get_r_targets_list(self):
        return self.__dict_to_list(self.r_targets)

    def get_l_centers_list(self):
        data = self.__dict_to_list(self.l_centers)
        return data

    def get_r_centers_list(self):
        data = self.__dict_to_list(self.r_centers)
        return data

    def get_random_test_samples(self, nsamples, ntargets):
        s_target = {i:np.empty((0,3), dtype='float32') for i in range(ntargets)}
        s_left   = {i:np.empty((0,3), dtype='float32') for i in range(ntargets)}
        s_right  = {i:np.empty((0,3), dtype='float32') for i in range(ntargets)}
        distribution = [i for i in range(nsamples)]
        candidates = np.random.choice(distribution, 5, False)
        for t in self.targets.keys():
            s_target[t] = np.take(self.targets[t], candidates, axis=0)
            if len(self.l_centers[t]) > 0:
                s_left[t] = np.take(self.l_centers[t], candidates, axis=0)
                self.l_centers[t] = np.delete(self.l_centers[t], candidates, axis=0)
            if len(self.r_centers[t]) > 0:
                s_right[t] = np.take(self.r_centers[t], candidates, axis=0)
                self.r_centers[t] = np.delete(self.r_centers[t], candidates, axis=0)
            self.targets[t] = np.delete(self.targets[t], candidates, axis=0)
        return s_target, s_left, s_right            


    def append_session_data(self, l_gaze, r_gaze, l_raw, r_raw):
        self.l_sess.append(l_gaze)
        self.r_sess.append(r_gaze)
        self.l_raw.append(l_raw)
        self.r_raw.append(r_raw)

    
    # def store_calibration(self):
    #     print(">>> Storing calibration data, please wait...")
    #     path = self.__check_or_create_path('calibration')
    #     for k in self.targets.keys():
    #         perc = int(k/len(self.targets.keys()) * 100)
    #         print(">>> {}%...".format(perc), end="\r", flush=True)
    #         c1, c2 = self.target_list[k]
    #         prefix = str(c1) + "_" + str(c2) + "_"
    #         # np.savez_compressed(path+prefix+ "img_scene", self.t_imgs[k])
    #         # np.savez_compressed(path+prefix+ "img_leye", self.l_imgs[k])
    #         # np.savez_compressed(path+prefix+ "img_reye", self.r_imgs[k])
    #         np.savez_compressed(path+prefix+ "tgt", self.targets[k])
    #         if len(self.l_centers[k]) > 0:
    #             np.savez_compressed(path+prefix+"leye", self.l_centers[k])
    #         if len(self.r_centers[k]) > 0:
    #             np.savez_compressed(path+prefix+"reye", self.r_centers[k])
    #     print(">>> Calibration data saved.")


    def store_calib_debug(self):
        path = self.__check_or_create_path('debug')
        l_tgt_train, r_tgt_train, l_norm_train, r_norm_train = [],[],[],[]
        l_tgt_test, r_tgt_test, l_norm_test, r_norm_test = [],[],[],[]
        for k in self.l_targets.keys():
            if k < 16:
                l_tgt_train.append(self.l_targets[k])
                r_tgt_train.append(self.r_targets[k])
                l_norm_train.append(self.l_centers[k])
                r_norm_train.append(self.r_centers[k])
            else:
                l_tgt_test.append(self.l_targets[k])
                r_tgt_test.append(self.r_targets[k])
                l_norm_test.append(self.l_centers[k])
                r_norm_test.append(self.r_centers[k])
        np.savez_compressed(path+'_train_left_target', l_tgt_train)
        np.savez_compressed(path+'_train_right_target', r_tgt_train)
        np.savez_compressed(path+'_train_left_normal', l_norm_train)
        np.savez_compressed(path+'_train_right_normal', r_norm_train)
        np.savez_compressed(path+'_test_left_target', l_tgt_test)
        np.savez_compressed(path+'_test_right_target', r_tgt_test)
        np.savez_compressed(path+'_test_left_normal', l_norm_test)
        np.savez_compressed(path+'_test_right_normal', r_norm_test)

    
    def load_calib_debug(self, path):
        train_l_tgt = np.load(path+'_train_left_target.npz')
        train_r_tgt = np.load(path+'_train_right_target.npz')
        train_l_norm = np.load(path+'_train_left_normal.npz')
        train_r_norm = np.load(path+'_train_right_normal.npz')
        test_l_tgt = np.load(path+'_test_left_target.npz')
        test_r_tgt = np.load(path+'_test_right_target.npz')
        test_l_norm = np.load(path+'_test_left_normal.npz')
        test_r_norm = np.load(path+'_test_right_normal.npz')
        train_data = [train_l_tgt, train_r_tgt, train_l_norm, train_r_norm]
        test_data = [test_l_tgt, test_r_tgt, test_l_norm, test_r_norm]
        return train_data, test_data



    def store_session(self):
        if len(self.l_sess) > 0:        
            print(">>> Saving session...")
            path = self.__check_or_create_path('session')
            np.savez_compressed(path+'_left_gaze', self.l_sess)
            np.savez_compressed(path+'_right_gaze', self.r_sess)
            np.savez_compressed(path+'_left_eye', self.l_raw)
            np.savez_compressed(path+'_right_eye', self.r_raw)
            print('>>> Session saved.')


    def __check_or_create_path(self, spec):
        '''
        spec -> either 'calibration' or 'session'
        '''
        path = os.getcwd() + "/data/"+self.uid+"/"+spec+"/"
        os.makedirs(path)
        return path