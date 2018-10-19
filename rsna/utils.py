import os
import numpy as np
import logging
import tensorflow as tf
import pydicom
import PIL

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import cpu_count


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

TF_IMAGE_FORMAT = b"jpeg"


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bbox_convert_to_min_max(x, y, width, height, **kwargs):
    """
    """
    return dict(
        xmin=int(x),
        xmax=int(x + width),
        ymin=int(y),
        ymax=int(y + height)
    )


def _bbox_convert_to_min_size(xmin, ymin, xmax, ymax, **kwargs):
    """
    """
    return dict(
        xmin=int(xmin),
        width=int(xmax - xmin),
        ymin=int(ymin),
        height=int(ymax - ymin)
    )


def _fill_annotation_dict(row, annotation_dict):
    """
    """
    if row['label'] == 'high_density':
        dd = _bbox_convert_to_min_max(**row.to_dict())
        dd.update({'label': row['label'], 'label_idx': row['label_idx']})
        annotation_dict[row['patientId']].append(dd)


def get_annotation_dict(df_train_labels):
    """
    """
    annotation_dict = {frame_id: [] for frame_id in df_train_labels['patientId'].unique()}
    df_train_labels.apply(lambda row: _fill_annotation_dict(row, annotation_dict), axis=1)

    return annotation_dict


class Annot:

    def __init__(self, data: dict, frame_path=None):
        self._data = data.copy()
        self._frame_path = frame_path

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, item):
        self._data[item] = item

    def __contains__(self, item):
        return item in self._data

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def get_min_max_bbox(self, frame_id):
        """
        """
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        labels = []
        for frame_annot in self._data[frame_id]:
            xmins.append(frame_annot["xmin"])
            xmaxs.append(frame_annot["xmax"])
            ymins.append(frame_annot["ymin"])
            ymaxs.append(frame_annot["ymax"])
            labels.append(frame_annot["label"])

        return dict(
            xmin=xmins,
            xmax=xmaxs,
            ymin=ymins,
            ymax=ymaxs,
            labels=labels
        )

    def get_min_size_bbox(self, frame_id):
        """
        """
        xmins = []
        widths = []
        ymins = []
        heights = []
        labels = []
        for frame_annot in self._data[frame_id]:
            xmins.append(frame_annot["xmin"])
            widths.append(frame_annot["xmax"] - frame_annot["xmin"])
            ymins.append(frame_annot["ymin"])
            heights.append(frame_annot["ymax"] - frame_annot["ymin"])
            labels.append(frame_annot["label"])

        return dict(
            xmin=xmins,
            widths=widths,
            ymin=ymins,
            heights=heights,
            labels=labels
        )

    @staticmethod
    def get_image(frame_path, frame_id):
        """
        """
        image = Annot._read_dcm(os.path.join(frame_path, frame_id + '.dcm'))

        return image

    def _create_tf_example(self, frame_id, frame_path, frame_annots):
        """
        """
        frame_filepath = os.path.join(frame_path, frame_id + ".jpg")
        frame_filepath_encoded = frame_filepath.encode("utf-8")
        frame_id_encoded = frame_id.encode("utf-8")

        image = PIL.Image.open(frame_filepath)
        width, height = image.size

        with open(frame_filepath, "rb") as f:
            image_encoded = f.read()

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        class_labels = []
        class_idc = []
        for frame_annot in frame_annots:
            xmins.append(frame_annot["xmin"] / width)
            xmaxs.append(frame_annot["xmax"] / width)
            ymins.append(frame_annot["ymin"] / height)
            ymaxs.append(frame_annot["ymax"] / height)
            class_labels.append(frame_annot["label"])
            class_idc.append(frame_annot["label_idx"])

        class_labels_encoded = [str(class_label).encode("utf-8") for class_label in class_labels]

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": int64_feature(height),
                    "image/width": int64_feature(width),
                    "image/filename": bytes_feature(frame_filepath_encoded),
                    "image/source_id": bytes_feature(frame_id_encoded),
                    "image/encoded": bytes_feature(image_encoded),
                    "image/format": bytes_feature(TF_IMAGE_FORMAT),
                    "image/object/bbox/xmin": float_list_feature(xmins),
                    "image/object/bbox/xmax": float_list_feature(xmaxs),
                    "image/object/bbox/ymin": float_list_feature(ymins),
                    "image/object/bbox/ymax": float_list_feature(ymaxs),
                    "image/object/class/text": bytes_list_feature(class_labels_encoded),
                    "image/object/class/label": int64_list_feature(class_idc),
                }
            )
        )

        return tf_example

    def _create_tf_example_no_annots(self, frame_id):
        """
        """
        frame_filepath = os.path.join(self._frame_path, frame_id + ".jpg")
        frame_filepath_encoded = frame_filepath.encode("utf-8")
        frame_id_encoded = frame_id.encode("utf-8")

        image = PIL.Image.open(frame_filepath)
        width, height = image.size

        with open(frame_filepath, "rb") as f:
            image_encoded = f.read()

        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": int64_feature(height),
                    "image/width": int64_feature(width),
                    "image/filename": bytes_feature(frame_filepath_encoded),
                    "image/source_id": bytes_feature(frame_id_encoded),
                    "image/encoded": bytes_feature(image_encoded),
                    "image/format": bytes_feature(TF_IMAGE_FORMAT),
                }
            )
        )

        return tf_example

    def create_ml_set(self, frame_path, output_path, test_fraction=0.2):
        """
        """
        np.random.seed(123)

        self._frame_path = frame_path
        self._out_path = output_path

        frame_ids = list(self._data.keys())

        test_frame_ids = np.random.choice(frame_ids, size=round(len(frame_ids) * test_fraction), replace=False)
        train_frame_ids = [frame for frame in frame_ids if frame not in test_frame_ids]

        logger.info(f"creating test and train sets with ratio {test_fraction}")
        logger.info(f"train size: {len(train_frame_ids)}")
        logger.info(f"test size: {len(list(test_frame_ids))}")
        logger.info(f"output_path: {output_path}")

        self._write_tf_records(list(test_frame_ids), out_filepath=os.path.join(output_path, 'test.tfrec'))
        self._write_tf_records(train_frame_ids, out_filepath=os.path.join(output_path, 'train.tfrec'))

    def _prepare_tf_example_from_jpg(self, frame_id):
        """
        """
        frame_annots = self._data[frame_id]

        frame_path = os.path.join(self._frame_path)
        tf_example = self._create_tf_example(frame_id=frame_id,
                                             frame_path=frame_path,
                                             frame_annots=frame_annots)
        return tf_example

    def _write_tf_records(self, frame_ids, out_filepath):
        """
        """
        writer = tf.python_io.TFRecordWriter(out_filepath)


        with Pool(cpu_count()) as p:
            tf_examples = list(tqdm(p.imap(self._prepare_tf_example_from_jpg, frame_ids), total=len(frame_ids)))

        for tf_example in tf_examples:
            if tf_example:
                writer.write(tf_example.SerializeToString())
            else:
                logger.info(f"invalid frame.")

        writer.close()

    @staticmethod
    def _read_dcm(filepath):
        """
        """
        return pydicom.read_file(filepath).pixel_array
