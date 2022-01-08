#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
import cv2
import numpy as np
import pycocotools.mask as mask_utils
from loguru import logger
from tqdm import tqdm


import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, dataloader, 
        img_size, postprocess_cfg, 
        num_classes, testdev=False, 
        metric=["bbox"]
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        annType = ["bbox", "segm"]
        self.dataloader = dataloader
        self.img_size = img_size
        self.postprocess_cfg = postprocess_cfg
        self.num_classes = num_classes
        self.testdev = testdev
        if isinstance(metric, str):
            assert metric in annType
            metric = [metric]
        elif isinstance(metric, list) or isinstance(metric, tuple):
            for m in metric:
                assert m in annType
            metric = set(metric)

        self.metric = metric
    

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids, _) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                if hasattr(model, "postrpocess"):
                    outputs = model.postprocess(
                        outputs,  num_classes=self.num_classes, **self.postprocess_cfg
                    )
                else:
                    outputs = postprocess(
                        outputs, num_classes=self.num_classes, **self.postprocess_cfg
                    )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        # outputs == 3 means don't contains mask 
        for (bboxes, clses, scores, masks, img_h, img_w, img_id) in zip(
            *outputs, info_imgs[0], info_imgs[1], ids
        ):
            if (bboxes is None) and (masks is None):
                continue

            assert (clses is not None) and (scores is not None)
            clses = clses.cpu().numpy()
            scores = scores.cpu().numpy()

            if bboxes is not None:
                bboxes = bboxes.cpu().numpy()
                # preprocessing: resize
                scale = min(
                    self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
                )
                bboxes /= scale
                bboxes = xyxy2xywh(bboxes)

            if masks is not None:
                masks = masks.cpu().numpy()
                masks =cv2.resize(
                    masks.transpose(1, 2, 0).astype(np.uint8), 
                    (self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_LINEAR
                )
                if masks.ndim==2:
                    masks = masks[..., None]
                masks = masks[: img_h, : img_w]

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(clses[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].tolist(),
                    "score": scores[ind].item(),
                    "segmentation": [mask_utils.encode(
                        np.array(masks[..., ind, None], order='F',     
                        dtype=np.uint8))[0]
                    ],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")
            eval_stat = {}
            for metric in self.metric:
                logger.info(f"evaluating {metric}")
                cocoEval = COCOeval(cocoGt, cocoDt, metric)
                cocoEval.evaluate()
                cocoEval.accumulate()
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    cocoEval.summarize()
                eval_stat[metric] = [cocoEval.stats[0], cocoEval.stats[1]]
                info += redirect_string.getvalue()

            return eval_stat, info
        else:
            return 0, 0, info
