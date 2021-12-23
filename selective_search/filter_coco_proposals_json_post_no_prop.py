import json
import os
import pickle
from posixpath import split

from filters import filter_none, filter_ratio

# NOTE(Tianyi): representations of path need to be modified
json_path = '/Users/zhao/Downloads/SoCo-main/data/train.json'
json_path_post = '/Users/zhao/Downloads/SoCo-main/data/train_post.json'

# no_props_images: class_name, TODO(JiaKui Hu): use json
# no_props_images = open('/Users/zhao/Downloads/SoCo-main/data/train_no_props_images.txt').readlines()
coco_root_proposals = '/Users/zhao/Downloads/SoCo-main/data'

split = 'train'

# NOTE(Tianyi): save as json
filtered_proposal_dict = {}
dd = 0
class_names = sorted(os.listdir(coco_root_proposals))
for ci, class_name in enumerate(class_names):
    filenames = sorted(os.listdir(os.path.join(coco_root_proposals, class_name)))
    dd += len(filenames)
    for fi, filename in enumerate(filenames):
        base_filename = os.path.splitext(filename)[0]
        cur_img_pro_path = os.path.join(coco_root_proposals, class_name, filename)

        with open(cur_img_pro_path, 'rb') as f:
            cur_img_proposal = pickle.load(f)
            g = []
            for region in cur_img_proposal:
                # NOTE(Tianyi): 2 bounding boxes
                cur_rect_1 = region['bbox1']
                cur_rect_2 = region['bbox2']
                try:
                    filtered_proposal_dict[base_filename].append(cur_rect_1)
                except KeyError as e:
                    filtered_proposal_dict[base_filename] = []
                    filtered_proposal_dict[base_filename].append(cur_rect_1)
                filtered_proposal_dict[base_filename].append(cur_rect_2)
            
            # NOTE(Tianyi): 没过filter了，需要的话再改
            props_size_ratio = filter_ratio(cur_img_proposal, r=3)
            # print(len(props_size_ratio))
            props_none = filter_none(cur_img_proposal)

            # if len(props_none) > 0:
            #     filtered_proposal_dict[base_filename] = props_none
            # elif len(props_size_ratio) > 0:
            #     filtered_proposal_dict[base_filename] = props_size_ratio
            

with open(json_path, 'w') as f:
    json.dump(filtered_proposal_dict, f)

print(dd)

# for no_props_image in no_props_images:
#     filename = no_props_image.strip()
#     class_name = filename.split('_')[0]
#     # NOTE(JiaKui Hu): like "./data/train/person/file_name.pkl", it contains dict
#     # TODO(JiaKui Hu): decode it(also in dataset)
#     """
#     regions : array of dict
#     [
#         {
#             'bbox_truth': (left, top, width, height),
#             'bbox1': ...,
#             'bbox2': ...,
#         },
#         ...
#     ]
#     """
#     cur_img_pro_path = os.path.join(coco_root_proposals, split, class_name, filename+'.pkl')

#     with open(cur_img_pro_path, 'rb') as f:
#         cur_img_proposal = pickle.load(f)
#         props_size_ratio = filter_ratio(cur_img_proposal['regions'], r=3)
#         props_none = filter_none(cur_img_proposal['regions'])
#         print("props_size_ratio", len(props_size_ratio))
#         print("props_none", len(props_none))
#         if len(props_size_ratio) > 0:
#             json_dict[filename] = props_size_ratio
#         elif len(props_none) > 0:
#             json_dict[filename] = filter_none(cur_img_proposal['regions'])


# with open(json_path_post, 'w') as f:
#     json.dump(json_dict, f)
