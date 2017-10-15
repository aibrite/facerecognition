from opencv.cascade.cascadebase import HaarCascadeBase
from opencv.cascade.downloadbase import DownloadPath


cascadeBase = HaarCascadeBase('downloads')


if __name__ == '__main__':
    # positive_choice = DownloadPath.get_user_request('Prepare positive images?')
    # if positive_choice is 'Yes':
    #     cascadeBase.prepare_positives()
    # else:
    #     print('Positive image preparation cancelled by user.')

    # download_choice = DownloadPath.get_user_request(
    #     'Downlaod negative images?')
    # if download_choice is 'Yes':
    #     cascadeBase.prepare_negatives(neg_urls=['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04105893',
    #                                             'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00015388'])
    # else:
    #     print('Negative image download cancelled by user.')

    # cascadeBase.remove_uglies()
    # cascadeBase.create_desc_files()
    # cascadeBase.form_positive_vector()
    # cascadeBase.train_classifier()

    cascadeBase.display_faces('cascadedata/data/cascade.xml')
