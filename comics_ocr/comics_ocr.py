import sys

# source: https://github.com/artefactory/NLPretext
from nlpretext import Preprocessor
from nlpretext.basic.preprocess import (normalize_whitespace, remove_eol_characters, lower_text,
                                        fix_bad_unicode)
from nltk.tokenize import WordPunctTokenizer

# for suppressing warnings
from comics_ocr.text_extractor import TextExtractor


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

sys.path.append('../')


class ComicsOCR:
    def __init__(self,
                 ocr_detector_config="/scratch/users/gsoykan20/projects/mmocr/work_dirs/fcenet_r50dcnv2_fpn_1500e_ctw1500_custom/fcenet_r50dcnv2_fpn_1500e_ctw1500_custom.py",
                 ocr_detector_checkpoint='/scratch/users/gsoykan20/projects/mmocr/work_dirs/fcenet_r50dcnv2_fpn_1500e_ctw1500_custom/best_0_hmean-iou:hmean_epoch_5.pth',
                 recog_config='/scratch/users/gsoykan20/projects/mmocr/work_dirs/master_custom_dataset/master_custom_dataset.py',
                 ocr_recognition_checkpoint='/scratch/users/gsoykan20/projects/mmocr/work_dirs/master_custom_dataset/best_0_1-N.E.D_epoch_4.pth',
                 det='FCE_CTW_DCNv2',
                 recog='MASTER'):
        self.text_extractor, self.text_preprocessor = ComicsOCR.set_text_processors(
            ocr_detector_config,
            ocr_detector_checkpoint,
            recog_config,
            ocr_recognition_checkpoint,
            det,
            recog)

    def extract_text(self, img_path: str):
        text = self.text_extractor.extract_text(img_path)
        preprocessed_text = self.text_preprocessor.run(text)
        sanitized_text = ComicsOCR.sanitize_text(preprocessed_text, self.text_preprocessor)
        return text, preprocessed_text, sanitized_text

    @staticmethod
    def sanitize_text(text, text_preprocessor) -> str:
        punc_tokenizer = WordPunctTokenizer()
        return ' '.join(punc_tokenizer.tokenize(text_preprocessor.run(text)))

    @staticmethod
    def set_text_processors(
            ocr_detector_config="/scratch/users/gsoykan20/projects/mmocr/work_dirs/fcenet_r50dcnv2_fpn_1500e_ctw1500_custom/fcenet_r50dcnv2_fpn_1500e_ctw1500_custom.py",
            ocr_detector_checkpoint='/scratch/users/gsoykan20/projects/mmocr/work_dirs/fcenet_r50dcnv2_fpn_1500e_ctw1500_custom/best_0_hmean-iou:hmean_epoch_5.pth',
            recog_config='/scratch/users/gsoykan20/projects/mmocr/work_dirs/master_custom_dataset/master_custom_dataset.py',
            ocr_recognition_checkpoint='/scratch/users/gsoykan20/projects/mmocr/work_dirs/master_custom_dataset/best_0_1-N.E.D_epoch_4.pth',
            det='FCE_CTW_DCNv2',
            recog='MASTER',
    ):
        text_preprocessor = ComicsOCR.get_minimal_text_preprocessor()
        text_extractor = TextExtractor(batch_mode=True,
                                       det=det,
                                       det_ckpt=ocr_detector_checkpoint,
                                       det_config=ocr_detector_config,
                                       recog=recog,
                                       recog_ckpt=ocr_recognition_checkpoint,
                                       recog_config=recog_config)
        return text_extractor, text_preprocessor

    @staticmethod
    def get_minimal_text_preprocessor():
        preprocessor = Preprocessor()
        preprocessor.pipe(lower_text)
        preprocessor.pipe(remove_eol_characters)
        preprocessor.pipe(normalize_whitespace)
        preprocessor.pipe(fix_bad_unicode)
        return preprocessor
