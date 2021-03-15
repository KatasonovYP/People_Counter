import imutils
from mark_detector import *
from people_counter import *
from FaceFinder import FaceFinder
from constants import *
'''
example for parser:
python main.py -w Data/models/yolo/v4-tiny.weights -f Data/models/yolo/v4-tiny.cfg -i Data/video/exit_2_out_2.mp4 -n Data/models/yolo/coco.names
'''


if __name__ == '__main__':

    args = parser()
    skip_frames = 30
    confidence = 0.4
    totalFrames = 0
    inside = 0
    trackers = []
    trackableObjects = {}
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    ct = CentroidTracker(maxDisappeared=40)

    arg = parser()
    video = load_video(arg['input'])
    model = load_model(arg['weights'], arg['cfg'])
    CLASSES = load_class_names(arg['names'])

    ff = FaceFinder(paths_to_photo_directory=arg['faces'])

    fps = FPS().start()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # start_video_from(video, 500)

    while True:
        grab, frame = video.read()

        if not grab:
            break

        # frame = imutils.rotate(frame, angle=20)
        # frame = frame[150:500, 1100:]
        frame = imutils.resize(frame, width=700)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        (H, W) = frame.shape[:2]
        h = H
        w = W // 4 * 3
        status = ''
        image = ff.find_faces(frame)
        if totalFrames % skip_frames == 0:
            rects, trackers = detect_objects(frame, model)
        elif totalFrames % skip_frames == skip_frames - 1:
            status = 'Detecting'
        else:
            if trackers:
                status = 'Tracking'
            else:
                status = 'Waiting'
            rects = track_objects(trackers, frame)

        objects = ct.update(rects)
        for (objectID, centroid) in objects.items():

            inside += update_count(trackableObjects, objectID, centroid, w)
            # draw_centroid(frame, objectID, w, h, centroid)

        fix_rects = []

        for rect in rects:
            fix_rect = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
            fix_rects.append(fix_rect)

        frame = filter_crops(frame, fix_rects)

        blk = np.zeros(frame.shape, np.uint8)
        cv.rectangle(blk, (0, h), (w, 0), (0, 255, 0), cv.FILLED)
        frame = cv.addWeighted(frame, 1.0, blk, 0.1, 1)
        cv.rectangle(blk, (w, h), (W, 0), (0, 0, 255), cv.FILLED)
        frame = cv.addWeighted(frame, 1.0, blk, 0.1, 1)

        frame = draw_persons(frame, fix_rects)
        frame = draw_info(frame, inside, status, H)

        cv2.line(frame, (w, 0), (w, h), (0, 0, 255), 2)

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
