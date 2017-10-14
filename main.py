from opencv.cascade.cascadebase import HaarCascadeBase
from opencv.cascade.downloadbase import DownloadPath


cascadeBase = HaarCascadeBase('downloads')


if __name__ == '__main__':
    positive_choice = DownloadPath.get_user_request('Prepare positive images?')
    if positive_choice is 'Yes':
        cascadeBase.prepare_positives()
    else:
        print('Positive image preparation cancelled by user.')
