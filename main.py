from opencv.cascade.cascadebase import HaarCascadeBase
from opencv.cascade.downloadbase import DownloadPath
from opencv.helper.prompt import Prompt


cascadeBase = HaarCascadeBase('downloads')


if __name__ == '__main__':
    # prepare_positives = Prompt.get_user_request('Prepare positive images?')
    # if prepare_positives:
    #     cascadeBase.prepare_positives()
    # else:
    #     print('Positive image preparation cancelled by user.')

    # cascadeBase.prepare_negatives(clean_false_links=True, neg_urls=['http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n00015388', 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09287968', 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n12992868', 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00017222'], bg_urls=[
    #     'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04105893'])

    # cascadeBase.remove_uglies()
    # cascadeBase.create_desc_files()
    # cascadeBase.create_positive_samples(
    #     file_name='info', positives_to_generate=50, maxxangle=0.5,
    #     maxyangle=-0.5, maxzangle=0.5)
    # cascadeBase.copy_generated_pos()
    cascadeBase.train_classifier(output_dir='cascadedata/data', vec_name='positives',
                                 num_stages=15, vec_width=20, vec_height=20, width=20, height=20)

    test_prompt = Prompt.get_user_request(
        'Start test with generated cascade file?')
    if test_prompt:
        cascadeBase.display_faces('cascadedata/data/cascade.xml')

    save_prompt = Prompt.get_user_request('Save generated cascade file?')
    if save_prompt:
        cascadeBase.save_cascade_file()
