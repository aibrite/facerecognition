import os
import datetime
from shutil import copy2


class DownloadDirs():
    def __init__(self, download_dir):
        self.main = download_dir
        self.pos = os.path.join(self.main, 'pos')
        self.neg = os.path.join(self.main, 'neg')
        self.uglies = os.path.join(self.main, 'uglies')
        self.bg_folder = 'img/bg'
        self.link_dir = 'links'
        self._check_directories()

    def get_sub_dirs(self):
        return [self.pos, self.neg, self.uglies]

    def _check_directories(self):
        for folder in self.get_sub_dirs():
            if not os.path.exists(folder):
                os.makedirs(folder)

        if not os.path.exists(self.bg_folder):
            os.makedirs(self.bg_folder)

        if not os.path.exists(self.link_dir):
            os.makedirs(self.link_dir)


class CascadeDirs():
    def __init__(self, cascade_dir='cascadedata'):
        self.main = cascade_dir
        self.data = os.path.join(self.main, 'data')
        self.info = os.path.join(self.main, 'info')
        self.pos = os.path.join(self.main, 'pos')
        self.cascade_save_dir = os.path.join(self.main, 'saved')
        self._check_directories()

    def get_sub_dirs(self):
        return [self.data, self.info, self.pos]

    def _check_directories(self):
        for folder in self.get_sub_dirs():
            if not os.path.exists(folder):
                os.makedirs(folder)

        if not os.path.exists(self.cascade_save_dir):
            os.makedirs(self.cascade_save_dir)

    def save_cascade_file(self):
        cascade_file = ''
        now = datetime.datetime.now()
        cascade_id = str(now.date()) + '__' + str(now.hour) + \
            '-' + str(now.minute) + '-' + str(now.second)

        os.makedirs(os.path.join(self.cascade_save_dir, cascade_id))
        os.makedirs(os.path.join(self.cascade_save_dir, cascade_id, 'stages'))

        save_dir = os.path.join(self.cascade_save_dir, cascade_id)

        for cas_file in os.listdir(self.data):
            file_path = os.path.join(self.data, cas_file)
            if os.path.splitext(cas_file)[0] == 'cascade':
                copy2(file_path, save_dir)
            else:
                copy2(file_path, os.path.join(save_dir, 'stages'))
