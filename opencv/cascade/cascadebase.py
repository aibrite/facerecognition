import cv2
import numpy as np
import os
import subprocess
from opencv.cascade.downloadbase import CascadeImageProcessor


class HaarCascadeBase(CascadeImageProcessor):

    def __init__(self, download_dir='downloads', cascade_dir='cascadedata'):
        super().__init__(download_dir=download_dir)
        self.cascade_dir = cascade_dir
        self.positive_file_count = 0
        self.info_file = ''

        if not os.path.exists(os.path.join(self.cascade_dir, 'info')):
            os.makedirs(os.path.join(self.cascade_dir, 'info'))

    def printVideoMessage(self, message='', key_message=''):
        if message == '':
            print('Starting Video Feed...')
            print('Press ESC to quit')
        else:
            print(message)
            print(key_message)

    def loadCascadeFile(self, cascade_file):

        if type(cascade_file) is list:
            cascades = []
            for cascade in cascade_file:
                cas = cv2.CascadeClassifier(cascade)
                cascades.append(cas)
            return cascades
        elif type(cascade_file) is str:
            cas = cv2.CascadeClassifier(cascade_file)
            return cas

    def displayEyesAndFaces(self, cascade_files, videoSource=0):
        cap = cv2.VideoCapture(videoSource)
        self.printVideoMessage()

        cascades = self.loadCascadeFile(cascade_files)
        face_cascade = cascades[0]
        eye_cascade = cascades[1]

        while True:
            _, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Original Video Feed', frame)

            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                eye_region_gray = gray_frame[y: y + h, x:x + w]
                eye_region_color = frame[y: y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(eye_region_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(eye_region_color, (ex, ey),
                                  (ex + ew, eh + eh), (0, 255, 0), 2)

            cv2.imshow('Faces and Eyes', frame)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def display_profile_face(self, cascade_file, videoSource=0):
        cap = cv2.VideoCapture(videoSource)
        self.printVideoMessage()

        profile_cascade = self.loadCascadeFile(cascade_file)

        while True:
            _, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Original Video Feed', frame)

            profiles = profile_cascade.detectMultiScale(gray_frame, 1.3, 5)
            for (x, y, w, h) in profiles:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv2.imshow('Face Profile', frame)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def form_positive_images(self, file_name='info', maxxangle=0.5, maxyangle=-0.5, maxzangle=0.5):
        file_count = len(os.walk(self.dirs['neg']).__next__()[2])
        positives_to_generate = file_count - 50
        self.positive_file_count = positives_to_generate

        print('Total negative files: {}'.format(file_count))
        print('Creating positive images for {} negatives...'.format(
            positives_to_generate))

        info_file = os.path.join(
            self.cascade_dir, 'info', file_name + '.lst')
        self.info_file = info_file
        output_dir = os.path.join(self.cascade_dir, 'info')

        subprocess.call(
            'opencv_createsamples -img downloads/pos/watch5050.jpg -bg bg.txt -info {0} -pngoutput {1} -maxxangle {2} -maxyangle {3} -maxzangle {4} -num {5}'.format(info_file, output_dir, maxxangle, maxyangle, maxzangle, positives_to_generate), shell=True)

    def form_positive_vector(self, file_name='positives', width=20, height=20):
        vector_file = os.path.join(self.cascade_dir, file_name + '.vec')
        print('Creating positive vector file...')
        subprocess.call(
            'opencv_createsamples -info {0} -num {1} -w {2} -h {3} -vec {4}'.format(self.info_file, self.positive_file_count, width, height, vector_file), shell=True)
