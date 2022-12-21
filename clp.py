import os
import re
import sys
import requests

#unknown error########################
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#for connecting sever
from socket import *
import pymysql

from numpy.lib.function_base import average
# 상위 디렉토리 패키지 가져오기
p = os.path.abspath('..')
sys.path.insert(1, p)

import numpy as np
import matplotlib.pyplot as plt
import time
#yyyymmdd to str
from time import ctime

# image resizing
import imutils

# for yolov4 detection
from ctypes import *
import random
import cv2
import time
import darknet
import argparse
from threading import Thread
from queue import Queue
# for DeepSORT
import core.utils as utils
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import threading

#for txt to send server
global detect_time 
detect_time = [" "]
global txt
txt = "     "

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default="./data/test3/test_vid2.mp4",
                        help="video source. If empty, uses ./data/uni_down.mp4")
    parser.add_argument("--output", type=str, default="",
                        help="inference video name. auto save if empty")
    parser.add_argument("--weights", default="./yolo_files/yolov4-tiny-3l-v3_best.weights", ##yolov4-tiny-3l-v2
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true', default=True,
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="./yolo_files/yolov4-tiny-3l-v3.cfg", ##yolov4-tiny-3l-v2
                        help="path to config file")
    parser.add_argument("--data_file", default="./yolo_files/obj-v2.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.4,
                        help="remove detections with confidence below this value")
    parser.add_argument("--size", type=int, default=608,
                        help="resize images to")  
    parser.add_argument("--count", type=bool, default=False,
                        help="count objects being tracked on screen")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

def set_saved_video(input_video, output_video, size):
    #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height

def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted

def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping

def video_capture(frame_queue, darknet_image_queue):
    isROIset = False
    while cap.isOpened():
        ret, frame = cap.read()
        if isROIset is False:
            img = frame
            # setROI(img)
            isROIset = True
        if not ret:
            break

        # 관심영역 추출.
        roi = frame[ROI_ymin:ROI_ymax, ROI_xmin:ROI_xmax] 
        #roi = cv2.fillConvexPoly(roi, nonROI, (0,0,0))
        ROI_frame = imutils.resize(roi.copy(), width = int((ROI_xmax - ROI_xmin)/resizing_rate))

        frame_rgb = cv2.cvtColor(ROI_frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame)
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        images = (img_for_detect, ROI_frame)
        darknet_image_queue.put(images)

        #off 
        time.sleep(0.001)
    cap.release()

def inference(darknet_image_queue, detections_queue, fps_queue, frame_num_queue, cap):
    max_fps = 0
    min_fps = 10000
    total_fps = 0.0
    average_fps = 0.0
    xmax = 0.0
    ymax = 0.0
    frame_num = 0
    while cap.isOpened():
        darknet_image, frame = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        
        detections_queue.put(detections)
        darknet.free_image(darknet_image)
        
        # 여기서 딥소팅을 하자.
        bboxes = []
        scores = []

        # yolov4의 BBOX, confidence 값 추출
        for label, confidence, bbox in detections:
            scores.append(confidence)
            bboxes.append(convert2original(frame, bbox))

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        # original_h, original_w, _ = frame.shape
        # print(frame.shape)
        # bboxes = utils.format_yolov4_boxes(bboxes, original_h, original_w)

        # by default allow all classes in .names file
        #allowed_classes = class_names
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(len(detections)):
            class_name = detections[i][0]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)

        # detected objects count & show
        if args.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        sort_detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
        # run non-maxima supression
        boxs = np.array([d.tlwh for d in sort_detections])
        for i in range(0,len(boxs)):
            if xmax < boxs[i][0] + boxs[i][2]:
                xmax = boxs[i][0] + boxs[i][2]
                # print(str(boxs[i][0]) + '    ' + str(boxs[i][1]) + '    ' + str(boxs[i][2]) + '    ' + str(boxs[i][3]))
                boxs[i][2] = video_width - boxs[i][0]
            if ymax < boxs[i][1] + boxs[i][3]:
                ymax = boxs[i][1] + boxs[i][3]

        # print("xmax : " + str(xmax) + "    ymax : " + str(ymax))
        scores = np.array([d.confidence for d in sort_detections])
        classes = np.array([d.class_name for d in sort_detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        sort_detections = [sort_detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(sort_detections, video_height)

        tracker_queue.put(tracker)
        bboxes_queue.put(boxs)

        # frame rate
        frame_num +=1
        fps = int(1/(time.time() - prev_time))
        if max_fps < fps:
            max_fps = fps
        if min_fps > fps:
            min_fps = fps
        fps_queue.put(fps)
        total_fps += float(fps)
        frame_num_queue.put(frame_num)
        average_fps = total_fps/frame_num
        print("Frame #: {}      FPS: {}      AVG_FPS: {:.3f}        MAX_FPS:  {}        MIN_FPS:  {}"
                .format(frame_num, fps, average_fps, max_fps, min_fps))
        
        # # if enable info flag then print details about each track
        #   if FLAGS.info:
        #    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        
        #off 
        time.sleep(0.001)
    cap.release()

def drawing(frame_queue, detections_queue, fps_queue, frame_num_queue, ):
    # traffic tracking용 변수
    trafficOut = 0
    trafficIn = 0

    person_in = 0
    child_in = 0

    person_out = 0
    child_out = 0    

    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.output, (video_width, video_height))
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        frame_num = frame_num_queue.get()
        tracker = tracker_queue.get()
        bboxes = bboxes_queue.get()
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        if frame is not None:
            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 

                bbox = track.to_tlbr()
                x1, y1, x2, y2 = bbox

                x1 *= resizing_rate; x2 *= resizing_rate
                y1 *= resizing_rate; y2 *= resizing_rate
                x1 += ROI_xmin;      x2 += ROI_xmin
                y1 += ROI_ymin;      y2 += ROI_ymin
                
                w = x2 - x1
                h = y2 - y1

                bbox = [int(round(x1 - (w / 2))), int(round(y1 - (h / 2))), 
                        int(round(x2 - (w / 2))), int(round(y2 - (h / 2)))]
                class_name = track.get_class()

                c1 = (int(bbox[0]) + int(bbox[2]))/2
                c2 = (int(bbox[1]) + int(bbox[3]))/2
                centerPoint = (int(c1), int(c2))

                # DOWN traffic count
                print(str(track.track_id) + "   |   " + str(track.sta))
                if (250 > int(bbox[1]+h) > 225) and track.noConsider == False and track.sta == 0:######
                    print("In bbox[1] : " + str(bbox[1]+h/2))
                    trafficIn += 1
                    track.sta = 1
                    if class_name == 'person':
                        person_in += 1
                        txt = time.strftime('%H:%M:%S', time.localtime())
                        txt = re.sub(r'[^0-9]', '', "0" + txt)
                        print(txt)

                    elif class_name == 'child':
                        child_out += 1         
                        txt = time.strftime('%H:%M:%S', time.localtime())
                        txt = re.sub(r'[^0-9]', '', "0" + txt)
                        print(txt)           
                        
                    track.noConsider = True
                    #cv2.line(frame, (0, video_height // 2 -40), (455, video_height // 2 -40), (0, 0, 0), 2)

                #UP traffic count
                if (255 < int(bbox[1]+h) < 275) and track.noConsider == False and track.sta == 0:######
                    print("On bbox[1] : " + str(bbox[1]+h/2))
                    print("track_id : " + str(track.track_id))
                    trafficOut += 1
                    track.sta = 1
                    if class_name == 'person':
                        person_out += 1
                        txt = time.strftime('%H:%M:%S', time.localtime())
                        txt = re.sub(r'[^0-9]', '', "1" + txt)
                        print(txt)
                        # send_txt()
                    elif class_name == 'child':
                        child_out += 1
                        txt = time.strftime('%H:%M:%S', time.localtime())
                        txt = re.sub(r'[^0-9]', '', "1" + txt)
                        print(txt)
                    
                    track.noConsider = True
                    cv2.line(frame, (0, video_height // 2 -40), (455, video_height // 2 -40), (0, 255, 0), 2)

                # draw bbox on screen
                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]
                text_color = (255,255,255)

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-20)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*13, int(bbox[1])+5), color, -1)
                cv2.circle(frame, (int(bbox[0]+w/2), int(bbox[1]+h)), 9, color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1]-5)),0, 0.5, text_color,1)

            # draw ROI 영역
            cv2.rectangle(frame, (ROI_xmin, ROI_ymin - 60), (ROI_xmin + 110, ROI_ymin), (233,250,50), -1)
            cv2.putText(frame, 'ROI', (ROI_xmin, ROI_ymin - 10), 0, 2, (0,0,0), 2)
            cv2.rectangle(frame, (ROI_xmin, ROI_ymin), (ROI_xmax, ROI_ymax), (233,250,50), 1)

            #xy test line
            cv2.line(frame, (ROI_xmin,225), (ROI_xmax,225), (0,0,255), 1)
            cv2.line(frame, (ROI_xmin,255), (ROI_xmax,255), (255,0,0), 1)
            print("video width : "+ str(video_width))
            print("video height : "+ str(video_height))

            #test for txt parameter
            try:
                txt = str(detect_time.pop(0))
                #cv2.putText(frame. txt, (800, video_height - ((i * 30) + 600)))
                #print("here!!!!" + txt)
            except:
                #cv2.putText(frame. txt, (800, video_height - ((i * 30) + 600)))
                #print("here!!!!" + txt)      
                pass
            
            info = [
                # ('traffic Count In', trafficIn),
                ('traffic Count', trafficOut - trafficIn),
                # ('traffic Count Out', trafficOut), 
            ] 

            in_type_info = [
                ('person', person_in),
                ('child', child_in)         
            ]

            out_type_info = [
                 ('person', person_out),
                 ('child', child_out)                 
            ]

            # 통행량
            for( i, (k, v)) in enumerate(info): #add code 
                text = "{}: {}".format(k, v) #add code
                cv2.putText(frame, text, (10, video_height - ((i * 30) + 600)), # (i * x) x부분을 변경하면 글자 높이 간격을 조절할 수 있음
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # DOWN
            for( i, (k, v)) in enumerate(in_type_info): #add code 
                text = "{}: {}".format(k, v) #add code
                cv2.putText(frame, "Down " + text, (100, video_height - ((i * 30) + 100)), # (i * x) x부분을 변경하면 글자 높이 간격을 조절할 수 있음
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # UP
            for( i, (k, v)) in enumerate(out_type_info): #add code[]
                text = "{}: {}".format(k, v) #add code
                cv2.putText(frame, "Up " + text, (video_width - 250, video_height - (((i+3) * 30) + 600)), # (i * x) x부분을 변경하면 글자 높이를 간격을 조절할 수 있음
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(frame, 'fps : ' + str(fps), (10,50), 0, 2, (0,255,0), 3)

            if video_height >= 720 and video_width >= 1280:
                show = cv2.resize(frame, (1280, 720))
            else:
                show = frame
            if not args.dont_show:
                cv2.imshow('Inference', show)
            if args.output is not None:
                video.write(frame)
            if cv2.waitKey(fps) == 27:
                break

        # if frame is not None:
        #     for label, confidence, bbox in detections:
        #         bbox_adjusted = convert2original(frame, bbox)
        #         detections_adjusted.append((str(label), confidence, bbox_adjusted))
        #     image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
        #     if not args.dont_show:
        #         cv2.imshow('Inference', image)
        #     if args.output is not None:
        #         video.write(image)
        #     if cv2.waitKey(fps) == 27:
        #         break

        #off 
       # time.sleep(0.001)
    cap.release()
    video.release()
    cv2.destroyAllWindows()

def to_server(serverName, serverPort):

    clientSocket = socket(AF_INET, SOCK_STREAM)
    clientSocket.connect((serverName,serverPort))

    reSize = clientSocket.recv(1024)
    reSize = reSize.decode()

    #can't find file
    if reSize == "error":
        return
    
    msg = "ready"
    clientSocket.sendall(msg.encode())

    with open(fileName,'w', encoding = "UTF-8") as f:
        #recv to file resize 
        data = clientSocket.recv(int(reSize))
        f.wirte(data.decode())

        
    clientSocket.close()
    return 0

#.txt send to server 
def send_txt(detect_class, detect_date):
    #for connecting mysql
    sql_host = "54.180.163.7"
    sql_port = "3306"
    sql_db = "Capstone"
    sql_user = "admin"
    sql_password ="baramdmf123"
    try:
        conn = pymysql.connect(sql_host, 
            user=sql_user, 
            passwd = sql_password, 
            db=sql_db, 
            use_unicode=True, 
            charset='utf8'
        )
        cursor = conn.cursor()
    except:
        sys.exit(1)
        pass
    
    query = "INSERT INTO test_table (detect_id, detect_date) VALUE (%d, %s))"#VALUE - CLASS, DATE
    val = (detect_class, detect_date)
    cursor.exeute(query, val)
    conn.commit()

    conn.close()
    return 0

if __name__ == '__main__':
    #for connecting server
    serverName = 'tmp' #ip
    serverPort = 21 #port

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
   
    ############unknown error###################### GPU exclusiv possession
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    nms_max_overlap = 1.0
    max_cosine_distance = 0.4
    nn_budget = None
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    # 트랙커를 통해 물체 정보(bbox, object_id)
    tracker_queue = Queue(maxsize=1)
    # bbox들 을 전달하는 큐
    bboxes_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    frame_num_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
    input_size = args.size
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)

    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ROI 영역 수동 지정
    #ROI_xmin = 250; ROI_xmax = 1500; ROI_ymin = 250; ROI_ymax = 900 ##lib
    #ROI_xmin = 250; ROI_xmax = 1500; ROI_ymin = 50; ROI_ymax = 900 ##uni
    #ROI_xmin = 0; ROI_xmax = video_width; ROI_ymin = 0; ROI_ymax = video_height ##all
    ROI_xmin = 700; ROI_xmax = 1150; ROI_ymin = 100; ROI_ymax = 1000 ##test3/down2.mp4
    

    # 영상 축소 수치. ex) 2 -> 1/2. 50% 축소
    resizing_rate = 2

    # 영상 재생 빈도. 영상의 종류에 따라 변경해주어야한다.
    frame_rate = 60
    frame_num = 0

    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue, frame_num_queue, cap)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue, frame_num_queue)).start()
