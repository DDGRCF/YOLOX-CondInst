#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.coco import maskUtils

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        with_mask=False,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.imgs = None
        self.masks = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.with_mask = with_mask
        self.annotations = self._load_coco_annotations()
        
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.data_dir + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                if self.with_mask and len(obj["segmentation"]) == 0:
                    continue
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        bbox_res = np.zeros((num_objs, 5))

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        resized_info = (int(height * r), int(width * r))
        img_info = (height, width)
        mask_res = []
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            bbox_res[ix, 0:4] = obj["clean_bbox"]
            bbox_res[ix, 4] = cls
            if self.with_mask:
                mask_res.append(obj['segmentation'])
                # mask_res.append(self._seg2np(obj["segmentation"], scale_ratio=r))
        # mask_res List: (#polygons, 4) -2: sub patch id -1: object id
        bbox_res[:, :4] *= r
        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )
        return (bbox_res, 
                img_info, 
                resized_info, 
                file_name, 
                mask_res if self.with_mask else None)
    

    def load_anno(self, index):
        return self.annotations[index][0]

    # def _seg2np(self, mask_ann, scale_ratio=None):
    #     mask_nplist = [np.asarray(m, dtype=np.float32) * 1. \
    #         if scale_ratio else scale_ratio for m in mask_ann]
    #     return mask_nplist

    def _single_load_mask(self, mask_ann, img_shape, scale_ratio=None):
        img_h, img_w = img_shape[:2]
        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        else:
            rle = mask_ann
        mask = maskUtils.decode(rle)
        if scale_ratio is not None:
            mask = cv2.resize(
                    mask,
                    (scale_ratio[1], scale_ratio[0]),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.uint8)
        return mask

    def load_resized_mask(self, mask_ann, img_shape,  scale_ratio, is_cat=False):
        masks = [self._single_load_mask(m, img_shape, scale_ratio) \
            for m in mask_ann]
        if is_cat:
            if len(masks): 
                masks = np.stack(masks, axis=-1)
            else:
                masks = np.empty((scale_ratio[0], scale_ratio[1], 0), dtype=np.uint8)
        return masks

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        bbox_res, img_info, resized_info, _, mask_res = self.annotations[index]

        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
        
        if mask_res is not None:
            mask_res = self.load_resized_mask(mask_res, img_info, resized_info, True)

        return (img, 
                bbox_res.copy(), 
                img_info, 
                np.array([id_]), 
                mask_res.copy() if self.with_mask else None)

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, bbox_target, img_info, img_id, mask_target = self.pull_item(index)

        if self.preproc is not None:
            img, bbox_target, mask_target = self.preproc(img, bbox_target, mask_target, self.input_dim)

        return img, bbox_target, img_info, img_id, mask_target

