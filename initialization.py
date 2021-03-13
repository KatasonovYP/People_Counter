import cv2 as cv
import numpy as np
import argparse
import time


def parser():
    """
    parser = argparse.ArgumentParser(description='Code for tracking people in the IT polygon.')
    parser.add_argument('-w', '--weights', default='Data/models/yolo/v4-tiny.cfg',
                        help='path to weights of model')
    parser.add_argument('-f', '--cfg', default='Data/models/yolo/v4-tiny.weights',
                        help='path to config of model')
    parser.add_argument('-n', '--names', type=str,
                        help='class names for model')
    parser.add_argument('-i', '--input', type=str,
                        help='path to optional input video file')
    parser.add_argument('-o', '--output', type=str,
                        help='path to optional output video file')
    parser.add_argument('-c', '--confidence', type=float, default=0.4,
                        help='minimum probability to filter weak detections')
    parser.add_argument('-s', '--skip-frames', type=int, default=30,
                        help='# of skip frames between detections')
    arguments = vars(parser.parse_args())
    """

    arguments = {
        'weights': 'Data/models/yolo/v4-tiny.weights',
        'cfg': 'Data/models/yolo/v4-tiny.cfg',
        'names': 'Data/models/yolo/coco.names',
        'input': 'Data/video/entry_2_in_1.mp4'
    }

    return arguments


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names


def load_model(path_weights, path_cfg):
    print('[INFO] loading model...')
    net = cv.dnn.readNet(path_weights, path_cfg)
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    model = cv.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255)
    print('[INFO] Done')
    return model


def load_video(path):
    print('[INFO] Starting video stream...')
    stream = cv.VideoCapture(path)
    time.sleep(2.0)
    print('[INFO] Done')
    return stream


def start_video_from(input_video, frames_skip):
    pos = input_video.get(cv.CAP_PROP_POS_FRAMES)
    input_video.set(cv.CAP_PROP_POS_FRAMES, pos + frames_skip)
    return input_video
