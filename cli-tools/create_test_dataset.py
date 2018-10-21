#!/usr/bin/env python3
import logging
import os
import click
import tensorflow as tf

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count

import sys
sys.path.append('../')

from rsna import utils


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s %(levelname)s "
                           "[%(module)s/%(funcName)s]: %(message)s")


@click.command()
@click.option("--frame_path", default="/mirco-kaggle/rsna/stage_1_test_images_jpg_eqhist/")
@click.option("--output_path", default="/home/ubuntu/kaggle-rsna/ml/v100/data/test/")
@click.option("--output_filename", default="unseen.tfrec")
def main(frame_path, output_path, output_filename):
    """
    """
    assert os.path.exists(frame_path), "raw data path not found."

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    frame_ids = [
        frame[:-4] for frame in os.listdir(frame_path) if frame.endswith(".jpg")
    ]

    logger.info(f"Creating tfrec file with no annotations for {len(frame_ids)} images")

    writer = tf.python_io.TFRecordWriter(os.path.join(output_path, output_filename))

    with Pool(cpu_count()) as p:
        tf_examples = list(tqdm(p.imap(
            utils.Annot({}, frame_path=frame_path)._create_tf_example_no_annots, frame_ids),
            total=len(frame_ids)
        ))

    for tf_example in tf_examples:
        if tf_example:
            writer.write(tf_example.SerializeToString())
        else:
            logger.info(f"invalid frame.")

    writer.close()

    logger.info(f"Done.")

if __name__ == "__main__":
    main()
