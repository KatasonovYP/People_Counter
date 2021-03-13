import imutils
from constants import *

'''
example for parser:
python main.py -w Data/models/yolo/v4-tiny.weights -f Data/models/yolo/v4-tiny.cfg -i Data/video/exit_2_out_2.mp4 /
-n Data/models/yolo/coco.names
'''


def crop_peoples(image, person_boxes):
    person_crops = []
    for box in person_boxes:
        startX = box[0]
        startY = box[1]
        endX = box[0] + box[2]
        endY = box[1] + box[3]

        person_crops.append(image[startY:endY, startX:endX])
    return person_crops


def draw_persons(image, person_boxes):
    person_boxes = list(person_boxes)
    for person_box in person_boxes:
        # color = COLORS[1 % len(COLORS)]
        # label = "%s : %f" % (CLASS_NAMES[classID[0]], score)
        blk = np.zeros(image.shape, np.uint8)
        cv.rectangle(blk, person_box, (255, 255, 255), cv.FILLED)
        image = cv.addWeighted(image, 1.0, blk, 0.25, 1)

    return image


def detect_shape(c):
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.04 * peri, True)

    if len(approx) == 3:
        shape = "triangle"

    elif len(approx) == 4:
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)

        shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"

    elif len(approx) == 5:
        shape = "pentagon"

    else:
        (x, y, w, h) = cv.boundingRect(approx)
        ar = w / float(h)
        shape = "circle" if 0.95 <= ar <= 1.05 else "oval"

    return shape


def filter_crops(image, person_boxes):
    person_crops = crop_peoples(image, person_boxes)
    arg = {'down': (50, 70, 73), 'up': (75, 255, 255), 'blur': (1, 1), 'erode': 0, 'dilate': 0}
    arg_2 = {'down': (105, 134, 72), 'up': (123, 255, 255), 'blur': (1, 1), 'erode': 0, 'dilate': 0}

    for (box, crop, i) in zip(person_boxes, person_crops, range(len(person_crops))):
        if crop.shape[0] * crop.shape[1] > 1:
            text = 'unknown'
            hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
            mask = cv.inRange(hsv, arg['down'], arg['up'])
            mask = cv.blur(mask, arg['blur'])
            contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

            if contours:
                contour = contours[0]
                max_area = 0
                for j in range(len(contours)):
                    area = cv.contourArea(contours[j])
                    if area > max_area:
                        contour = contours[j]
                        max_area = area

                (x, y, w, h) = cv.boundingRect(contour)
                mark = hsv[y:y + h, x:x + w]

                mark = cv.inRange(mark, arg_2['down'], arg_2['up'])

                result = cv.countNonZero(mark)
                if result:
                    text = 'blue'
                else:
                    text = 'black'

                blk = np.zeros(crop.shape, np.uint8)
                cv.rectangle(blk, (x, y), (x + w, y + h), (0, 255, 0), cv.FILLED)
                crop = cv.addWeighted(crop, 1.0, blk, 0.7, 1)
                image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = crop

            x_text, y_text = box[0], box[1]
            blk = np.zeros(image.shape, np.uint8)
            cv.rectangle(blk, (x_text, y_text - 10), (x_text + 70, y_text), (255, 255, 255), cv.FILLED)
            image = cv.addWeighted(image, 1.0, blk, 0.9, 1)
            cv.putText(image, text, (x_text, y_text), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return image


if __name__ == '__main__':

    args = parser()
    video = load_video(args['input'])
    model = load_model(args['weights'], args['cfg'])
    while True:
        grabbed, frame = video.read()

        if not grabbed:
            break

        frame = imutils.resize(frame, width=500)
        person_rects = []
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for (classID, score, box) in zip(classes, scores, boxes):
            if CLASS_NAMES[classID[0]] == 'person':
                person_rects.append(box)

        cv.imshow('start', frame)

        filter_crops(frame, person_rects)
        frame = draw_persons(frame, person_rects)

        cv.imshow('detections', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cv.destroyAllWindows()
