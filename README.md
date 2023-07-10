to build locally after cloning

```shell
pip install comics-ocr[cuda] -f https://download.pytorch.org/whl/torch_stable.html

or 

pip install comics-ocr[cpu]
```

You can get the necessary model checkpoints and configs from
[COMICS TEXT+](https://github.com/gsoykan/comics_text_plus) repository.

## Usage

```python
# Import library
from comics_ocr import ComicsOCR

# initalize the model
e2e_ocr_model = ComicsOCR(
    ocr_detector_config="fcenet_r50dcnv2_fpn_1500e_ctw1500_custom/fcenet_r50dcnv2_fpn_1500e_ctw1500_custom.py",
    ocr_detector_checkpoint='fcenet_r50dcnv2_fpn_1500e_ctw1500_custom/best_0_hmean-iou:hmean_epoch_5.pth',
    recog_config='master_custom_dataset.py',
    ocr_recognition_checkpoint='best_0_1-N.E.D_epoch_4.pth',
    det='FCE_CTW_DCNv2',
    recog='MASTER')

# Run the model
img_path = "speech_bubble/0/3/9.jpg"
text, preprocessed_text, sanitized_text = e2e_ocr_model.extract_text(img_path)
```
