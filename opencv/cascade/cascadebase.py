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
        if not os.path.exists(os.path.join(self.cascade_dir, 'data')):
            os.makedirs(os.path.join(self.cascade_dir, 'data'))

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

    def display_faces(self, cascade_files, videoSource=0):
        cap = cv2.VideoCapture(videoSource)
        self.printVideoMessage()

        face_cascade = self.loadCascadeFile(cascade_files)
        # cascades = self.loadCascadeFile(cascade_files)
        # face_cascade = cascades[0]
        # eye_cascade = cascades[1]

        while True:
            _, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Original Video Feed', frame)

            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            cv2.imshow('Faces', frame)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def create_desc_files(self):
        if os.path.exists('bg.txt'):
            os.remove('bg.txt')
        if os.path.exists('info.dat'):
            os.remove('info.dat')

        for sign_type in os.listdir(self.dirs['main']):
            if sign_type != 'neg' and sign_type != 'pos':
                continue
            else:
                for img in os.listdir(self.dirs[sign_type]):
                    if sign_type == 'neg':
                        line = os.path.join(self.dirs[sign_type], img) + '\n'
                        with open('bg.txt', 'a') as f:
                            f.write(line)

                    elif sign_type == 'pos':
                        line = os.path.join(
                            self.dirs[sign_type], img) + ' 1 0 0 50 50\n'
                        with open('info.dat', 'a') as f:
                            f.write(line)

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

    def form_positive_vector(self, file_name, samples, width, height):
        vector_file = os.path.join(self.cascade_dir, file_name + '.vec')
        self.vector_file = vector_file
        print('Creating positive vector file...')
        # subprocess.call(
        #     'opencv_createsamples -info {0} -num {1} -w {2} -h {3} -vec {4}'.format('info.dat', self.positive_file_count, width, height, vector_file), shell=True)
        subprocess.call(
            'opencv_createsamples -info {0} -num {1} -w {2} -h {3} -vec {4}'.format('info.dat', samples, width, height, vector_file), shell=True)

    def train_classifier(self, output_dir='cascadedata/data', vec_name='positives', num_stages=10, vec_width=20, vec_height=20, width=20, height=20):
        # cascade_file = os.path.join(self.cascade_dir, file_name + '.vec')
        self.form_positive_vector(
            vec_name, vec_samples, width=vec_width, height=vec_height)

        vec_samples = len(os.walk(self.dirs['pos']).__next__()[2]) - 50

        num_pos = len(os.walk(self.dirs['pos']).__next__()[2]) * 0.8
        num_neg = num_pos / 2
        print('Training Cascade Classifier...')
        subprocess.call(
            'opencv_traincascade -data {0} -vec {1} -bg bg.txt -numPos {2} -numNeg {3} -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numStages {4} -w {5} -h {6}'.format(output_dir, self.vector_file, num_pos, num_neg, num_stages, width, height), shell=True)
