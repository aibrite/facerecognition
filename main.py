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
    #     cascadeBase.prepare_negatives(neg_urls=['http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00015388'], bg_urls=[
    #                                   'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04105893'])
    # else:
    #     print('Negative image download cancelled by user.')

    # cascadeBase.remove_uglies()
    # cascadeBase.create_desc_files()
    # cascadeBase.create_positive_samples(
    # file_name='info', positives_to_generate=50, maxxangle=0.5,
    # maxyangle=-0.5, maxzangle=0.5)
    cascadeBase.join_info_files()
    # cascadeBase.form_positive_vector()
    # cascadeBase.train_classifier(output_dir='cascadedata/data', vec_name='positives',
    # num_stages=10, vec_width=20, vec_height=20, width=20, height=20)

    # cascadeBase.display_faces('cascadedata/data/cascade.xml')
