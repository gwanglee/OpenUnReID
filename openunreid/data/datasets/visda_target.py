# Written by Yixiao Ge

import glob
import os.path as osp
import re
import warnings

from ..utils.base_dataset import ImageDataset


class Visda(ImageDataset):
    """PersonX
    Reference:
        Sun et al. Dissecting Person Re-identification from the Viewpoint of Viewpoint.
            CVPR 2019.
    URL: `<https://github.com/sxzrt/Instructions-of-the-PersonX-dataset#a-more-chanllenging-subset-of-personx>`  # noqa

    Dataset statistics:
    # identities: 1266 (train + query)
    # images: 9840 (train) + 5136 (query) + 30816 (gallery)
    """

    dataset_dir = "challenge_datasets"
    dataset_url = (
        "https://drive.google.com/file/d/1hiHoDt3u7_GfeICMdEBt2Of8vXr1RF-U/view"
    )
    dataset_url_gid = "1hiHoDt3u7_GfeICMdEBt2Of8vXr1RF-U"  # download from this gd ID

    def __init__(self, root, mode, val_split=0.2, del_labels=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels
        # self.download_dataset(
        #     self.dataset_dir, self.dataset_url, dataset_url_gid=self.dataset_url_gid
        # )
        assert (val_split > 0.0) and (
            val_split < 1.0
        ), "the percentage of val_set should be within (0.0,1.0)"

        # allow alternative directory structure
        dataset_dir = osp.join(self.dataset_dir, "subset1")
        if osp.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            warnings.warn(
                "The current data structure is deprecated. Please "
                'put data folders such as "bounding_box_train" under '
                '"subset1".'
            )

        subsets_cfgs = {
            "train": (
                osp.join(self.dataset_dir, "target_training/image_train"),
                [0.0, 1.0 - val_split],
                True,
            ),
            "val": (
                osp.join(self.dataset_dir, "target_training/image_train"),
                [1.0 - val_split, 1.0],
                False,
            ),
            "trainval": (
                osp.join(self.dataset_dir, "target_training/image_train"),
                [0.0, 1.0],
                True,
            ),
            "query": (osp.join(self.dataset_dir, "target_validation/image_query"), [0.0, 1.0], False),
            "gallery": (
                osp.join(self.dataset_dir, "target_validation/image_gallery"),
                [0.0, 1.0],
                False,
            ),
        }
        try:
            cfgs = subsets_cfgs[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode. Got {}, but expected to be "
                "one of [train | val | trainval | query | gallery]".format(self.mode)
            )

        required_files = [self.dataset_dir, cfgs[0]]
        self.check_before_run(required_files)

        data = self.process_dir(*cfgs)
        super(Visda, self).__init__(data, mode, **kwargs)

    def process_dir(self, dir_path, data_range, relabel=False):

        data = []

        if 'validation' in dir_path:
            pardir = osp.dirname(dir_path)
            idx_path = osp.join(pardir, 'index_validation_query') if ('query' in dir_path) \
                    else osp.join(pardir, 'index_validation_gallery')

            with open(idx_path, 'r') as rf:
                lines = rf.readlines()
        else:
            pardir = osp.dirname(dir_path)
            idx_path = osp.join(pardir, 'label_target_training.txt')

            with open(idx_path, 'r') as rf:
                lines = rf.readlines()

        for l in lines:
            cur = l.strip().split(' ')

            filename = cur[0]
            cid = int(cur[1])
            pid = int(cur[2]) if len(cur) > 3 else 0

            data.append((osp.join(dir_path, filename), pid, cid))

        print(dir_path)
        for d in data[:3]:
            print(d)

        return data
