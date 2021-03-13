import cv2


def nothing(x):
    pass


def create_track_bars():
    cv2.namedWindow('border')
    cv2.namedWindow('filter')

    cv2.createTrackbar('min_hue', 'border', 0, 255, nothing)
    cv2.createTrackbar('max_hue', 'border', 255, 255, nothing)

    cv2.createTrackbar('min_saturation', 'border', 0, 255, nothing)
    cv2.createTrackbar('max_saturation', 'border', 255, 255, nothing)

    cv2.createTrackbar('min_value', 'border', 0, 255, nothing)
    cv2.createTrackbar('max_value', 'border', 255, 255, nothing)

    cv2.createTrackbar('blur', 'filter', 1, 5, nothing)
    cv2.createTrackbar('erode', 'filter', 0, 5, nothing)
    cv2.createTrackbar('dilate', 'filter', 0, 5, nothing)


def get_borders():
    min_hue = cv2.getTrackbarPos('min_hue', 'border')
    max_hue = cv2.getTrackbarPos('max_hue', 'border')

    min_saturation = cv2.getTrackbarPos('min_saturation', 'border')
    max_saturation = cv2.getTrackbarPos('max_saturation', 'border')

    min_value = cv2.getTrackbarPos('min_value', 'border')
    max_value = cv2.getTrackbarPos('max_value', 'border')

    blur = cv2.getTrackbarPos('blur', 'filter')
    if blur is 0:
        blur = 1

    parameters = {
        'down': (min_hue, min_saturation, min_value),
        'up': (max_hue, max_saturation, max_value),
        'blur': (blur, blur),
        'erode': cv2.getTrackbarPos('erode', 'filter'),
        'dilate': cv2.getTrackbarPos('dilate', 'filter')
    }

    return parameters


if __name__ == '__main__':

    cam = cv2.VideoCapture(0)
    create_track_bars()

    while True:
        _, frame = cam.read()
        param = get_borders()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(frame, param['down'], param['up'])
        cv2.imshow('Video', frame)
        cv2.imshow('Mask', mask)

        if cv2.waitKey(1) == ord('q'):
            print(param)
            break


cv2.destroyAllWindows()
