#!/usr/bin/env python3
import logging
import os
import pandas as pd
import click
import random

import sys
sys.path.append('../')

from rsna import utils


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s %(levelname)s "
                           "[%(module)s/%(funcName)s]: %(message)s")


@click.command()
@click.option("--limit", default=None)
@click.option("--only_target_images", default=False)
@click.option("--raw_data_path", default="/home/mhess/")
@click.option("--output_path", default="/home/mhess/kaggle-rsna/ml/v100/data/")
@click.option("--test_fraction", default=0.2)
@click.option("--label_map_filepath", default=None)
def main(limit, only_target_images, raw_data_path, output_path, test_fraction, label_map_filepath):
    """
    """
    assert os.path.exists(raw_data_path), "raw data path not found."

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_dicom_dir = os.path.join(raw_data_path, 'stage_1_train_images_jpg_dev')
    train_labels = os.path.join("/mirco-kaggle/rsna", 'stage_1_train_labels.csv')

    df_train_labels = pd.read_csv(train_labels)

    if not label_map_filepath:
        label_map = {0: 'other', 1: 'high_density'}


    df_train_labels = df_train_labels.assign(label_idx=df_train_labels['Target'])
    df_train_labels['label'] = df_train_labels['label_idx'].apply(lambda x: label_map[x])

    logger.info(f"Creating ML dataset ...")
    annotation_dict = utils.get_annotation_dict(df_train_labels=df_train_labels)

    logger.info(f"Balancing ...")
    empty_image_keys = [k for k, v in annotation_dict.items() if not v]
    non_empty_image_keys = [k for k, v in annotation_dict.items() if v]
    logger.info(f"images w/o annotations: {len(empty_image_keys)}")
    logger.info(f"images w/ annotations: {len(non_empty_image_keys)}")


    annotation_dict_empty_images = {
        k: annotation_dict[k] for k in random.sample(empty_image_keys, len(non_empty_image_keys))
    }
    annotation_dict_non_empty_images = {
        k: annotation_dict[k] for k in non_empty_image_keys
    }

    if limit:
        logger.info(f"limit set to: {limit}")
        annotation_dict_empty_images = dict(random.sample(annotation_dict_empty_images.items(), int(limit) // 2))
        annotation_dict_non_empty_images = dict(random.sample(annotation_dict_non_empty_images.items(), int(limit) // 2))

    logger.info(f"images w/o annotations: {len(annotation_dict_empty_images)}")
    logger.info(f"images w/ annotations: {len(annotation_dict_non_empty_images)}")

    if only_target_images:
        logger.info(f"using images w/ annotations")
        annotation_dict = annotation_dict_non_empty_images
    else:
        annotation_dict = {**annotation_dict_empty_images, **annotation_dict_non_empty_images}

    logger.info(f"FINAL dataset size: {len(annotation_dict)}")

    annot = utils.Annot(annotation_dict)
    annot.create_ml_set(frame_path=train_dicom_dir, output_path=output_path, test_fraction=test_fraction)
    logger.info(f"Done.")

if __name__ == "__main__":
    main()
