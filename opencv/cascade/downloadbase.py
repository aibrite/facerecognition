import urllib.request
import cv2
import numpy as np
import os
from shutil import copy2
from opencv.cascade.paths import DownloadDirs


class DownloadPath():

    def __init__(self, download_dir):

        self.download_dirs = DownloadDirs(download_dir)

        self._check_directories()

    def _check_directories(self):

        for folder in self.download_dirs.get_sub_dirs():
            if not os.path.exists(folder):
                os.makedirs(folder)

        if not os.path.exists(self.download_dirs.bg_folder):
            os.makedirs(self.download_dirs.bg_folder)

        if not os.path.exists(self.download_dirs.link_dir):
            os.makedirs(self.download_dirs.link_dir)

    def get_user_request(question):
        user_options = {
            'y': 'Yes',
            'n': 'No'
        }

        prompt = ''
        for key, value in user_options.items():
            line = 'Type {0} for {1}\n'.format(key, value)
            prompt += line

        user_selection = input(
            '{0}\n{1}'.format(question, prompt))

        return user_options.get(user_selection, 'n')


class CascadeImageProcessor(DownloadPath):

    def __init__(self, download_dir='downloads'):
        super().__init__(download_dir)

    def resize_image(self, image, is_neg=True):
        if is_neg is True:
            resized = cv2.resize(image, (100, 100))
        else:
            resized = cv2.resize(image, (50, 50))
        return resized

    def grayscale_and_save(self, file_name):
        gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        resized = self.resize_image(gray)
        cv2.imwrite(file_name, resized)
        print('Resized Grayscale Image Saved')

    def download_and_process(self, clean, count=None, raw_neg=False):
        pic_count = count + 1
        base_url = ''
        link_file = ''

        if raw_neg is False:
            base_url = self.download_dirs.neg
            link_file = os.path.join(
                self.download_dirs.link_dir, 'negative_urls.txt')
        else:
            base_url = self.download_dirs.bg_folder
            link_file = os.path.join(
                self.download_dirs.link_dir, 'background_urls.txt')

        urls_arr = []
        with open(link_file, encoding='utf-8') as f:
            urls = f.read()
            urls_arr = urls.splitlines()
            f.close()

        for image_url in urls_arr:
            try:
                print('Downloading Image No {}: {}'.format(pic_count, image_url))
                urllib.request.urlretrieve(
                    image_url, os.path.join(base_url, str(pic_count) + '.jpg'))
                self.grayscale_and_save(
                    os.path.join(base_url, str(pic_count) + '.jpg'))

                pic_count += 1

            except Exception as err:
                print(str(err))
                if clean is True:
                    with open(link_file, 'w', encoding='utf-8') as outfile:
                        for line in urls_arr:
                            if line != image_url:
                                outfile.write(line + '\n')
                            else:
                                urls_arr.remove(line)
                        outfile.close()

        return pic_count

    def prepare_negatives(self, clean_false_links=False, neg_urls=['http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00015388', 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09287968'], bg_urls=['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04105893']):

        last_neg = 0

        if not os.path.exists(os.path.join(self.download_dirs.link_dir, 'negative_urls.txt')):
            for neg_url in neg_urls:
                neg_image_url_list = urllib.request.urlopen(
                    neg_url).read().decode()
                for link in neg_image_url_list.split('\n'):
                    with open(os.path.join(self.download_dirs.link_dir, 'negative_urls.txt'), 'a', encoding='utf-8') as f:
                        f.write(link)
                        f.close()

        last_neg = self.download_and_process(
            clean_false_links, count=last_neg, raw_neg=False)

        if not os.path.exists(os.path.join(self.download_dirs.link_dir, 'background_urls.txt')) and os.path.exists(os.path.join(self.download_dirs.link_dir, 'negative_urls.txt')):
            last_neg = len(os.listdir(self.download_dirs.neg))
            for bg_url in bg_urls:
                bg_image_url_list = urllib.request.urlopen(
                    bg_url).read().decode()
                for link in bg_image_url_list.split('\n'):
                    with open(os.path.join(self.download_dirs.link_dir, 'background_urls.txt'), 'a', encoding='utf-8') as f:
                        f.write(link)
                        f.close()

        last_neg = self.download_and_process(
            clean_false_links, count=last_neg, raw_neg=True)

        for bg in os.listdir(self.download_dirs.bg_folder):
            copy2(bg, self.download_dirs.neg)

    def prepare_positives(self, positive_dir='img/raw_images'):
        print('Preparing positive images...')
        count = 1
        for img in sorted(os.listdir(positive_dir)):
            image = cv2.imread(os.path.join(
                positive_dir, img), cv2.IMREAD_COLOR)
            face_rect = image[50: 150, 50: 150]
            resized = self.resize_image(face_rect, is_neg=False)
            cv2.imwrite(os.path.join(
                self.download_dirs.pos, str(count) + '.jpg'), resized)
            count += 1

        print('Raw image preparation completed')

    def identify_uglies(self):
        question = 'Type the number of the ugly image:'
        cancel = 'Type cancel to exit'

        prompt = True
        while prompt is True:
            ugly_num = input('{}\n{}\n'.format(question, cancel))
            ugly_path = os.path.join(self.download_dirs.neg, ugly_num + '.jpg')

            if ugly_num == 'cancel':
                prompt = False
                return False
            elif os.path.exists(ugly_path):
                copy2(ugly_path, self.download_dirs.uglies)
                prompt = False
            elif not os.path.exists(ugly_path):
                print('No such image')
                prompt = True

        return True

    def remove_uglies(self):
        remove_request = self.identify_uglies()
        if remove_request is True:
            count = 0
            # for sign_path in os.listdir(self.dirs['main']):
            #     if sign_path == 'uglies':
            #         continue
            #     for img in os.listdir(self.dirs[sign_path]):
            #         for ugly in os.listdir(self.dirs['uglies']):

            #             try:
            #                 current_img_path = os.path.join(
            #                     self.dirs[sign_path], img)
            #                 ugly_img = cv2.imread(os.path.join(
            #                     self.dirs['uglies'], ugly))
            #                 current_img = cv2.imread(current_img_path)

            #                 if ugly_img.shape == current_img.shape and not (np.bitwise_xor(ugly_img, current_img).any()):
            #                     print('Ugly image found: {}'.format(
            #                         current_img_path))
            #                     print('Image removed')
            #                     os.remove(current_img_path)
            #                     count += 1

            #             except Exception as err:
            #                 print(str(err))

            for folder in [self.download_dirs.bg_folder, self.download_dirs.pos, self.download_dirs.neg]:
                for img in os.listdir(folder):
                    for ugly in os.listdir(self.download_dirs.uglies):
                        try:
                            current_img_path = os.path.join(folder, img)
                            ugly_img = cv2.imread(os.path.join(
                                self.download_dirs.uglies, ugly))
                            current_img = cv2.imread(current_img_path)

                            if ugly_img.shape == current_img.shape and not (np.bitwise_xor(ugly_img, current_img).any()):
                                print('Ugly image found: {}'.format(
                                    current_img_path))
                                print('Image removed')
                                os.remove(current_img_path)
                                count += 1

                        except Exception as err:
                            print(str(err))

            # for img in os.listdir(self.bg_folder):
            #     for ugly in os.listdir(self.dirs['uglies']):

            #         try:
            #             current_img_path = os.path.join(
            #                 self.bg_folder, img)
            #             ugly_img = cv2.imread(os.path.join(
            #                 self.dirs['uglies'], ugly))
            #             current_img = cv2.imread(current_img_path)

            #             if ugly_img.shape == current_img.shape and not (np.bitwise_xor(ugly_img, current_img).any()):
            #                 print('Ugly image found: {}'.format(
            #                     current_img_path))
            #                 print('Image removed')
            #                 os.remove(current_img_path)
            #                 count += 1

            #         except Exception as err:
            #             print(str(err))

            print(
                'Ugly images successfully removed. Total uglies removed: {0}'.format(count))
        else:
            print('Ugly image removal cancelled by user.')
