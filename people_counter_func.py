from CentroidTracker import CentroidTracker
from TrackableObject import TrackableObject
from constants import *
from imutils.video import FPS
import numpy as np
import imutils
import dlib
import cv2


def parser():
    arguments = {
        'weights': 'Data/models/yolo/v4.weights',
        'cfg': 'Data/models/yolo/v4.cfg',
        'names': 'Data/models/yolo/coco.names',
        'input': 'Data/video/pc.mp4'
    }

    return arguments


def detect_objects(image, current_model):
    new_trackers = []
    new_rectangles = []
    classes, scores, boxes = current_model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    for (classID, score, box) in zip(classes, scores, boxes):
        if CLASS_NAMES[classID[0]] == 'person':

            startX = box[0]
            startY = box[1]
            endX = box[0] + box[2]
            endY = box[1] + box[3]

            rect = dlib.rectangle(startX, startY, endX, endY)
            new_rectangles.append((startX, startY, endX, endY))
            tracker = dlib.correlation_tracker()
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            tracker.start_track(rgb, rect)
            new_trackers.append(tracker)

    return new_rectangles, new_trackers


def track_objects(current_trackers, image):
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    new_rectangles = []
    for tracker in current_trackers:

        tracker.update(rgb)
        pos = tracker.get_position()

        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())

        new_rectangles.append((startX, startY, endX, endY))
    return new_rectangles


def update_count(trackableObjects, objectID, current_centroid, border_line):
    count = 0
    to = trackableObjects.get(objectID, None)

    if to is None:
        to = TrackableObject(objectID, current_centroid)

    else:
        # y = [c[1] for c in to.centroids]
        x = [c[0] for c in to.centroids]
        direction = current_centroid[0] - np.mean(x)
        to.centroids.append(current_centroid)

        if not to.counted:

            if direction < 0 and current_centroid[0] < border_line:
                count += 1
                to.counted = True

            # elif direction > 0 and current_centroid[0] > border_line:
            #     count -= 1
            #     to.counted = True

    trackableObjects[objectID] = to

    return count


def draw_centroid(image, ID, centroid):
    text = f'ID {ID}'
    cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


def draw_info(image, current_count, current_status, height):
    info = [
        ('Inside', current_count),
        ('Status', current_status),
    ]

    blk = np.zeros(image.shape, np.uint8)
    cv.rectangle(blk, (10, 10), (175, 50), (255, 255, 255), cv.FILLED)
    image = cv.addWeighted(image, 1.0, blk, 0.9, 1)

    for (i, (k, v)) in enumerate(info):
        text = f'{k}: {v}'
        cv2.putText(image, text, (10, 5 + ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return image


if __name__ == '__main__':

    args = parser()
    skip_frames = 30
    totalFrames = 0
    inside = 0
    trackers = []
    trackableObjects = {}
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    ct = CentroidTracker(maxDisappeared=40)

    args = parser()
    video = load_video(args['input'])
    model = load_model(args['weights'], args['cfg'])
    CLASSES = load_class_names(args['names'])

    fps = FPS().start()

    start_video_from(video, 480)

    while True:
        _, frame = video.read()
        if frame is None:
            break

        frame = imutils.rotate(frame, angle=20)
        frame = frame[150:500, 1100:]
        # frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        (H, W) = frame.shape[:2]
        h = H
        w = W // 4 * 3

        status = 'Waiting'
        if totalFrames % skip_frames == 0:
            status = 'Detecting'
            rects, trackers = detect_objects(frame, model)
        else:
            rects = track_objects(trackers, frame)
            if rects:
                status = 'Tracking'

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():

            inside += update_count(trackableObjects, objectID, centroid, w)
            draw_centroid(frame, objectID, centroid)

        cv2.line(frame, (w, 0), (w, h), (0, 255, 255), 2)
        frame = draw_info(frame, inside, status, H)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

        totalFrames += 1
        fps.update()

    fps.stop()
    print(f'[INFO] elapsed time: {fps.elapsed()}')
    print(f'[INFO] approx. FPS: {fps.fps()}')
    print(f'[INFO] count persons: {inside}')

    video.release()
    cv2.destroyAllWindows()
