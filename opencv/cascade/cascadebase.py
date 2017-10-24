import cv2
import numpy as np
import os
import subprocess
from opencv.cascade.downloadbase import CascadeImageProcessor
from shutil import copy2


class HaarCascadeBase(CascadeImageProcessor):

    def __init__(self, download_dir='downloads', cascade_dir='cascadedata'):
        super().__init__(download_dir=download_dir)
        self.cascade_dir = cascade_dir
        self.positive_file_count = 0
        self.info_file = ''
        self.generated_pos_dir = os.path.join(self.cascade_dir, 'pos')

        self.cascade_dirs = {
            'main': self.cascade_dir,
            'data': os.path.join(self.cascade_dir, 'data'),
            'info': os.path.join(self.cascade_dir, 'info'),
            'pos': os.path.join(self.cascade_dir, 'pos')
        }

        for key, value in self.cascade_dirs.items():
            if not os.path.exists(self.cascade_dirs[key]):
                os.makedirs(self.cascade_dirs[key])

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
        if os.path.exists('bg_sample.txt'):
            os.remove('bg_sample.txt')

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

        for bg_sample in os.listdir(self.bg_folder):
            line = os.path.join(self.bg_folder, bg_sample) + '\n'
            with open('bg_sample.txt', 'a') as f:
                f.write(line)

    def create_random_bg_file(self, bg_count=50):
        if os.path.exists('bg_sample.txt'):
            os.remove('bg_sample.txt')

        count = 0
        for bg_sample in os.listdir(self.bg_folder):
            line = os.path.join(self.bg_folder, bg_sample) + '\n'
            with open('bg_sample.txt', 'a') as f:
                f.write(line)
            if count == bg_count * 2:
                break
            count += 1

    def create_positive_samples(self, file_name='info', positives_to_generate=50, maxxangle=0.5, maxyangle=-0.5, maxzangle=0.5):
        file_count = len(os.walk(self.bg_folder).__next__()[2])
        # positives_to_generate = file_count - 50
        # self.positive_file_count = positives_to_generate

        positives = []
        for pos in os.listdir(self.dirs['pos']):
            if int(os.path.splitext(pos)[0]) % 5 == 0:
                positives.append(pos)

        print('Total background files: {}'.format(file_count))
        print('Total positive files selected: {}'.format(len(positives)))
        print('Samples to generate for each positive: {}'.format(
            positives_to_generate))
        print('Generating positive samples...')

        pos_count = 0
        for pos in positives:
            info_file = os.path.join(
                self.cascade_dirs['pos'], str(pos_count), file_name + '.lst')
            output_dir = os.path.join(
                self.cascade_dirs['pos'], str(pos_count))
            os.makedirs(output_dir)
            pos_path = os.path.join(self.dirs['pos'], pos)
            self.create_random_bg_file(bg_count=positives_to_generate)
            subprocess.call('opencv_createsamples -img {0} -bg bg_sample.txt -info {1} -pngoutput {2} -maxxangle {3} -maxyangle {4} -maxzangle {5} -num {6}'.format(
                pos_path, info_file, output_dir, maxxangle, maxyangle, maxzangle, positives_to_generate), shell=True)
            pos_count += 1

    def copy_generated_pos(self, desired_num=4000):
        print('Reorganizing generated positives...')
        print('Copying and renaming generated positives...')
        count = 0
        for folder in sorted(os.listdir(self.cascade_dirs['pos'])):
            for pos_file in sorted(os.listdir(os.path.join(self.cascade_dirs['pos'], folder), key=lambda x: int(os.path.splitext(x)[0]))):
                if os.path.splitext(pos_file)[1] != '.lst':
                    img = cv2.imread(os.path.join(self.cascade_dirs['pos'], folder,
                                                  pos_file))
                    cv2.imwrite(os.path.join(
                        self.cascade_dirs['info'], str(count) + '.jpg'), img)
                    if count == desired_num:
                        break
                    count += 1

        print('Positives copied.')
        self.join_info_files(desired_num)

    def join_info_files(self, pos_num):
        print('Cascading info files...')
        info_files = []
        for folder in sorted(os.listdir(self.cascade_dirs['pos'])):
            for pos_file in sorted(os.listdir(os.path.join(self.cascade_dirs['pos'], folder), key=lambda x: int(os.path.splitext(x)[0]))):
                if os.path.splitext(pos_file)[1] == '.lst':
                    info_file = os.path.join(
                        self.cascade_dirs['pos'], folder, pos_file)
                    info_files.append(info_file)

        if os.path.exists(os.path.join(self.cascade_dirs['info'], 'info.lst')):
            os.remove(os.path.join(self.cascade_dirs['info'], 'info.lst'))

        line_count = 0
        with open(os.path.join(self.cascade_dirs['info'], 'info.lst'), 'a') as outfile:
            for info_file in info_files:
                with open(info_file) as infile:
                    for line in infile:
                        img = str(line_count)
                        line = img + line[line.find('.'):]
                        outfile.write(line)
                        if line_count == pos_num:
                            break
                        line_count += 1
        print('Generated positives reorganized.')

    def form_positive_vector(self, file_name, samples, width, height):
        vector_file = os.path.join(self.cascade_dir, file_name + '.vec')
        self.vector_file = vector_file
        print('Creating positive vector file...')

        subprocess.call(
            'opencv_createsamples -info {0} -num {1} -w {2} -h {3} -vec {4}'.format('info.lst', samples, width, height, vector_file), shell=True)

    def train_classifier(self, output_dir='cascade/data', vec_name='positives', num_stages=10, vec_width=20, vec_height=20, width=20, height=20):
        # cascade_file = os.path.join(self.cascade_dir, file_name + '.vec')
        total_pos = len(os.walk(os.path.join(
            self.cascade_dir, 'pos')).__next__()[2])
        vec_samples = 3500
        self.form_positive_vector(
            vec_name, vec_samples, width=vec_width, height=vec_height)

        # num_pos = total_pos * 0.4
        num_pos = 3000
        num_neg = 1400
        # num_neg = num_pos / 2

        print('Training Cascade Classifier...')
        subprocess.call(
            'opencv_traincascade -data {0} -vec {1} -bg bg.txt -numPos {2} -numNeg {3} -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numStages {4} -w {5} -h {6}'.format(output_dir, self.vector_file, num_pos, num_neg, num_stages, width, height), shell=True)
