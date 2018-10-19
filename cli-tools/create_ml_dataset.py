#!/usr/bin/env python3
import logging
import os
import pandas as pd
import click

import sys
sys.path.append('../')

from rsna import utils


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s %(levelname)s "
                           "[%(module)s/%(funcName)s]: %(message)s")


@click.command()
@click.option("--limit", default=None)
@click.option("--raw_data_path", default="/home/ubuntu/")
@click.option("--output_path", default="/home/ubuntu/kaggle-rsna/ml/v100/data/")
@click.option("--test_fraction", default=0.2)
@click.option("--label_map_filepath", default=None)
def main(limit, raw_data_path, output_path, test_fraction, label_map_filepath):
    """
    """
    assert os.path.exists(raw_data_path), "raw data path not found."

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_dicom_dir = os.path.join(raw_data_path, 'stage_1_train_images_jpg')
    train_labels = os.path.join("/mirco-kaggle/rsna", 'stage_1_train_labels.csv')

    df_train_labels = pd.read_csv(train_labels)

    if not label_map_filepath:
        label_map = {0: 'other', 1: 'high_density'}

    if limit:
        df_train_labels = df_train_labels.assign(label_idx=df_train_labels['Target']).head(int(limit))
    else:
        df_train_labels = df_train_labels.assign(label_idx=df_train_labels['Target'])
    df_train_labels['label'] = df_train_labels['label_idx'].apply(lambda x: label_map[x])

    logger.info(f"Creating ML dataset ...")
    annotation_dict = utils.get_annotation_dict(df_train_labels=df_train_labels)
    annot = utils.Annot(annotation_dict)
    annot.create_ml_set(frame_path=train_dicom_dir, output_path=output_path, test_fraction=test_fraction)
    logger.info(f"Done.")

if __name__ == "__main__":
    main()
