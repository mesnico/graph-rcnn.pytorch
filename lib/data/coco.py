from torchvision.datasets.coco import CocoCaptions
import os


class COCOVal(CocoCaptions):
    # We basically use the CocoCaptions dataset for loading COCO images

    def __init__(self, cfg, split="val", transforms=None):

        dataset_root_val = os.path.join(cfg.DATASET.PATH, '{}2014'.format(split))
        dataset_json_val = os.path.join(cfg.DATASET.PATH,
                                        'stanford_split_annots', 'captions_{}2014.json'.format(split))
        super().__init__(dataset_root_val, dataset_json_val)
        self.vg_transforms = transforms

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        img, _ = self.vg_transforms(img, None)
        return img, None, index
