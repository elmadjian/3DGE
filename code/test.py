import cv2
import sys
import uvc
import numpy as np
import eye
import eye_img_processor as eip
from multiprocessing import Process, Pipe
import eyefitter as ef
import geometry as geo
from threading import Thread
import os
import socket
import time

#testing cameras. Use test.py --uvc [cam_id]
if sys.argv[1] == "--uvc":
    cam = int(sys.argv[2])
    dev_list = uvc.device_list()
    print('devices found:')
    for d in dev_list:
        print('>>>', d)
    cap = uvc.Capture(dev_list[cam]['uid'])
    # #cap2 = uvc.Capture(dev_list[2]['uid'])
    print(sorted(cap.avaible_modes))
    cap.frame_mode = cap.avaible_modes[0]
    print("current mode:", cap.frame_mode)
    # #cap.bandwidth_factor = 1.3
    while True:
        frame = cap.get_frame()
     #   frame2 = cap2.get_frame()
        cv2.imshow('uvc test', frame.bgr)
      #  cv2.imshow('uvc test2', frame2.bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.close()
    cv2.destroyAllWindows()

#RECORDING
if sys.argv[1] == "--rec":
    dev_list = uvc.device_list()
    print(dev_list)
    cap = uvc.Capture(dev_list[3]['uid'])
    cap.frame_mode = (400,400,60)
    cap.bandwidth_factor = 1.3
    out = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 120, (400,400))
    while True:
        frame = cap.get_frame()
        out.write(frame.bgr)
        cv2.imshow('recording', frame.bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.close()
    out.release()
    cv2.destroyAllWindows()

#TESTING
if sys.argv[1] == "--track":
    cap = cv2.VideoCapture('test.avi')
    WIDTH  = 400
    HEIGHT = 400
    lut = np.empty((1,256), np.uint8)
    gamma = 0.65
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 0.01 * (WIDTH * HEIGHT)
    params.filterByCircularity = True
    params.minCircularity = 0.15
    params.filterByConvexity = True
    params.minConvexity = 0.8
    detector = cv2.SimpleBlobDetector_create(params)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i in range(256):
                lut[0,i] = np.clip(pow(i/255.0, gamma) *255.0, 0, 255)
            img = cv2.LUT(gray, lut)
            cv2.imshow("tracking", img)
            cv2.waitKey(5)

            #MAX_TREE
            keypoints = detector.detect(img)
            detected = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('keypoints', detected)         
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

if sys.argv[1] == "--eye":
    cap = cv2.VideoCapture('pupil.mp4')
    lut = np.empty((1,256), np.uint8)
    gamma = 0.65
    eyeobj = eye.EyeCamera()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            for i in range(256):
                lut[0,i] = np.clip(pow(i/255.0, gamma) *255.0, 0, 255)
            img = cv2.LUT(frame, lut)
            img, centroid = eyeobj.process(img)
            cv2.imshow('test', img)
            cv2.waitKey(1)


if sys.argv[1] == '--3D':
    #cap = cv2.VideoCapture('pupil.mp4')
    cap = cv2.VideoCapture('pupil2.mkv')
    #cap = cv2.VideoCapture('demo.mp4')
    #cap = cv2.VideoCapture('test.avi')
    #eyeobj = eip.EyeImageProcessor(0,(400,400),0,0,0,0)
    eyeobj = eip.EyeImageProcessor(0,(480,640),0,0,0,0)
    sensor_size = (3.6, 4.8) #mm
    focal_length = 6         #mm
    #fitter = ef.EyeFitter(focal_length, (400,400), sensor_size)
    fitter = ef.EyeFitter(focal_length, (480,640), sensor_size)
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img, ellipse = eyeobj.process(frame, mode_3D=True)               
            cv2.imshow('test', img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


if sys.argv[1] == '--simulate_HMD':
    ip, port = "127.0.0.1", 50022
    # if os.path.isfile('config/hmd_config.txt'):
    #     with open('config/hmd_config.txt', 'r') as hmd_config:
    #         data = hmd_config.readline()
    #         ip, port = data.split(':')
    #         port = int(port)
    socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    socket.bind((ip, 50021))
    target_list = []
    for y in np.linspace(0,1, 3):
        for x in np.linspace(0,1, 3):
            target_list.append(np.array([x,y,1], dtype=np.float32))
    seed = np.random.randint(0,99)
    rnd  = np.random.RandomState(seed)
    rnd.shuffle(target_list)
    target_list.append(np.array([0.5,0.4,0.6], dtype=np.float32))
    print('-- Connecting...')
    socket.sendto('C'.encode(), (ip,port))
    try:
        response = socket.recv(1024).decode()
        if response:
            print('-- Connected!')
        else:
            print('-- Remote host not found.')
            sys.exit()
    except Exception:
        print('-- Error trying to connect.')
        sys.exit()
    time.sleep(0.5)
    for t in target_list:
        msg = 'N:' + str(t[0])+':'+str(t[1])+':'+ str(t[2])
        socket.sendto(msg.encode(), (ip,port))
        print('-- Next target...')
        time.sleep(0.25)
        socket.sendto('R'.encode(), (ip,port))
        print('-- Recording target...')
        time.sleep(0.25)
    socket.sendto('F'.encode(), (ip,port))
    print('-- End calibration...')
    y = np.linspace(0.46, 0.54, 25)
    x_l = np.linspace(0.45, 0.495, 25)
    x_r = np.linspace(0.505, 0.55, 25) 
    count = 0
    port = 50023
    # socket.sendto('F'.encode(), (ip,port))
    # try:
    #     response = socket.recv(1024).decode()
    #     if response:
    #         print('-- Connected!')
    #     else:
    #         print('-- Remote host not found.')
    #         sys.exit()
    # except Exception:
    #     print('-- Error trying to connect.')
    #     sys.exit()
    while True:
        try:
            demand = socket.recv(1024).decode()
            if demand.startswith('G'):
                y1 = np.random.choice(y)
                y2 = np.random.choice(y)
                x1 = np.random.choice(x_l)
                x2 = np.random.choice(x_r)
                x1, y1, z1 = '{:.8f}'.format(x1),'{:.8f}'.format(y1),'{:.8f}'.format(1.0)
                x2, y2, z2 = '{:.8f}'.format(x2),'{:.8f}'.format(y2),'{:.8f}'.format(1.0)
                msg = 'G:'+x1+':'+y1+':'+z1+':'+x2+':'+y2+':'+z2
                socket.sendto(msg.encode(), (ip,port))
        except Exception as e:
            print("no request from HMD...", e)
            count += 1
            if count > 3:
                break

    




