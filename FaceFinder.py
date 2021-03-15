import face_recognition
import os
from constants import *


class FaceFinder:
    def __init__(self, paths_to_photo_directory):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

        files = os.listdir(paths_to_photo_directory)

        for name in files:
            if name[-4:] == '.jpg':
                path = paths_to_photo_directory + name
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)[0]

                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name[:-4])

    def find_faces(self, image):

        rgb_small_frame = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        if self.process_this_frame:
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        for (startY, endX, endY, startX), name in zip(self.face_locations, self.face_names):

            cv.rectangle(image, (startX, startY), (endX, endY), (255, 255, 255), 2)

            # cv.rectangle(image, (startX, endY - 35), (endX, endY), (255, 255, 255), cv.FILLED)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(image, name, (startX + 6, endY - 6), font, 0.5, (0, 0, 0), 2)

        return image


if __name__ == '__main__':

    video_capture = cv.VideoCapture(0)
    directory = 'Data/images/faces/'
    FF = FaceFinder(directory)

    while True:
        grab, frame = video_capture.read()

        if not grab:
            break

        frame = FF.find_faces(frame)

        cv.imshow('Video', frame)

        if cv.waitKey(1) == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()
