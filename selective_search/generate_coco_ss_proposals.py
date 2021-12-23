import os
import pickle
import random

import numpy as np
import cv2
import skimage
import PIL.Image as Image

from pycocotools.coco import COCO


random.seed(118)


def process_bbox(bbox: list) -> list:
    """
    perturbation (could be modified)
    :params bbox: List, (left, top, w, h)
    :return     : List, the same as input
    """
    # print(bbox)
    left, top, width, height = bbox[:4]
    left += random.random() * 5
    top += random.random() * 5
    width = width * (random.random() * 0.5 + 1)
    height = height * (random.random() * 0.5 + 1)
    return (left, top, width, height)


def process_classes(source_path, coco):
    """
    process one classes of COCO dataset, use known GT
    remove the selective search and random selection from SoCo, use known bbox
    :params category_id: category_id(see it in coco annotations)
    :params source_path: the path with images
    :params coco:        COCO form annotation_file
    """
    # get categories list
    categories = coco.loadCats(coco.getCatIds())

    fname_class = '/Users/zhao/Downloads/SoCo-main/data/ann.txt'
    f_class = []
    # NOTE(Tianyi): representations of path need to be modified
    for category_id in categories:
        print(f"Category: {category_id} started")
        catIds = coco.getCatIds(catNms=category_id["name"])
        imgIds = coco.getImgIds(catIds=catIds)
        os.makedirs('/Users/zhao/Downloads/SoCo-main/data/{}'.format(category_id["name"]))
        for i in range(len(imgIds)):
            # get img info
            img_info = coco.loadImgs(imgIds[i])[0]
            # get img annotations
            annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            # get img
            # img_path = os.path.join(source_path, img_info["file_name"])
            base_filename = os.path.splitext(img_info["file_name"])[0]
            # img = np.array(Image.open(img_path).convert('RGB'))
            tmp = []
            tmp.append(img_info["file_name"])
            tmp.append(str(catIds[0]))
            f_class.append(tmp)
            
            cur_img_proposal = []
            # NOTE(Tianyi): modified the saving structure as an array of dict
            """
            [
                {
                    'bbox': (left, top, width, height),
                    'bbox1': (left, top, width, height),
                    'bbox2': (left, top, width, height)
                },
                ...
            ]
            """
            for a in anns:
                d = {}
                d["bbox"] = a["bbox"]
                d["bbox1"] = process_bbox(a["bbox"])
                d["bbox2"] = process_bbox(a["bbox"])
                cur_img_proposal.append(d)
                # cv2.rectangle(
                #     img_show,
                #     (int(bbox1[0]), int(bbox1[1])),
                #     (int(bbox1[0])+int(bbox1[2]), int(bbox1[1])+int(bbox1[3])),
                #     color=(0, 0, 255), lineType=0
                # )
                # cv2.rectangle(
                #     img_show,
                #     (int(bbox2[0]), int(bbox2[1])),
                #     (int(bbox2[0])+int(bbox2[2]), int(bbox2[1])+int(bbox2[3])),
                #     color=(0, 255, 0), lineType=0
                # )
            
            # cur_img_proposal['bbox_truth'] = bbox
            # cur_img_proposal['bbox1'] = bbox1
            # cur_img_proposal['bbox2'] = bbox2
            # bbox == regions(in local SoCo)(float, not int)

            cur_img_pro_path = os.path.join('/Users/zhao/Downloads/SoCo-main/data', category_id["name"], base_filename+'.pkl')
            with open(cur_img_pro_path, 'wb') as f:
                pickle.dump(cur_img_proposal, f)

            # cv2.imshow("img", img_show)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # if i == 2:
                # break

        # print("processed class:", category_id)
        # break
    
    with open(fname_class, 'w') as f:
        for fc in f_class:
            f.write(fc[0] + '\t' + fc[1] + '\n')


if __name__ == "__main__":
    coco_root = ".."
    coco_year = "2017"
    coco_annotations_root = "annotations"

    split = "val"
    dataType = 'val2017'
    source_path = os.path.join(coco_root, coco_year, split)
    annotation_file = os.path.join(
        coco_root, coco_year, coco_annotations_root,
        "".join(["instances_", split, coco_year, ".json"])
    )
    annotation_file = '{}/annotations/instances_{}.json'.format(coco_root, dataType)
    print(source_path)

    coco = COCO(annotation_file)
    process_classes(source_path, coco)
