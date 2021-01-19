import matplotlib
matplotlib.use( 'tkagg' )

import matplotlib.pyplot as plt

import os
from collections import OrderedDict
import cv2
import numpy as np
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision import transforms
from scipy.ndimage.measurements import label

from ssd import build_ssd
from data import *
from torch.utils.data import Dataset, DataLoader
from utils import draw_boxes, helpers, save_boxes
import gtdb.feature_extractor


class ArgStub():

    def __init__ (self):
        self.cuda = False
        self.kernel = (1, 5)
        self.padding = (0, 2)
        self.phase = 'test'
        self.visual_threshold = 0.25
        self.verbose = False
        self.exp_name = 'SSD'
        self.model_type = 512
        self.use_char_info = False
        self.limit = -1
        self.cfg = 'hboxes512'
        self.batch_size = 1
        self.num_workers = 0
        self.neg_mining = True
        self.log_dir = 'logs'
        self.stride = 0.1
        self.window = 1200
        self.test_data = "testing_data"
        self.dataset_root = "/Users/ilhambintang/Latihan/riset/ScanSSD"
        self.save_folder = "/Users/ilhambintang/Latihan/riset/ScanSSD/eval"
        self.exp_name = "testing"



def draw_box (image, boxes):
    for b in boxes:
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)



def _img_to_tensor (image):
    rimg = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA).astype(np.float32)
    # width = image.shape[0]
    # height = image.shape[1]
    # max_width = 1024
    # coef = max_width/width
    # new_width = int(width * coef)
    # new_height = int(height * coef)
    # rimg = cv2.resize(image, (new_height, new_width), interpolation = cv2.INTER_AREA).astype(np.float32)

    rimg -= np.array((246, 246, 246), dtype=np.float32)
    rimg = rimg[:, :, (2, 1, 0)]
    return torch.from_numpy(rimg).permute(2, 0, 1)



def FixImgCoordinates (images, boxes):
    new_boxes = []
    if isinstance(images, list):
            for i in range(len(images)):
                print(images[i].shape)
                bbs = []
                for o_box in boxes[i] :
                    b = [None] * 4
                    b[0] = int(o_box[0] * images[i].shape[0])
                    b[1] = int(o_box[1] * images[i].shape[1])
                    b[2] = int(o_box[2] * images[i].shape[0])
                    b[3] = int(o_box[3] * images[i].shape[1])
                    bbs.append(b)

                new_boxes.append(bbs)
    else:
        bbs = []
        for o_box in boxes[0] :
            b = [None] * 4
            b[0] = int(o_box[0] * images.shape[0]) 
            b[1] = int(o_box[1] * images.shape[1])
            b[2] = int(o_box[2] * images.shape[0])
            b[3] = int(o_box[3] * images.shape[1])
            bbs.append(b)

        new_boxes.append(bbs)

    return new_boxes


def DrawAllBoxes(images, boxes):

    for i in range(len(images)):
        draw_box(images[i], boxes[i])


def convert_to_binary(image):
    try:
        print(image)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(gray_image)
    except Exception as e:
        print(e)

    im_bw = np.zeros(gray_image.shape)
    im_bw[gray_image > 127] = 0
    im_bw[gray_image <= 127] = 1

    return im_bw


class MathDetector():

    def __init__(self, weight_path, args):
        self.args = args
        net = build_ssd(args, 'test', config.exp_cfg[args.cfg], -1, args.model_type, 2)
        self._net = net # nn.DataParallel(net)
        weights = torch.load(weight_path, map_location = torch.device('cpu'))

        new_weights = OrderedDict()
        for k, v in weights.items():
            name = k[7:] # remove `module.`
            new_weights[name] = v

        self._net.load_state_dict(new_weights)
        self._net.eval()

        self.dataset = GTDBDetection(args, self.args.test_data, split='test',
                                transform=BaseTransform(self.args.model_type, (246, 246, 246)),
                                target_transform=GTDBAnnotationTransform())

        self.data_loader = DataLoader(self.dataset, self.args.batch_size,
                                 num_workers=self.args.num_workers,
                                 shuffle=False, collate_fn=detection_collate,
                                 pin_memory=True)

        self.boxes = []
        self.scores = []


    def Detect (self, thres, images):

        done = 0

        for batch_idx, (images, targets, metadata) in enumerate(self.data_loader):
            done = done + len(images)

            with torch.no_grad():
                images = Variable(images)
                targets = [Variable(ann) for ann in targets]

            y, debug_boxes, debug_scores = self._net(images)  # forward pass
            detections = y.data

            k = 0
            for img, meta in zip(images, metadata):

                img_id = meta[0]
                x_l = meta[1]
                y_l = meta[2]

                img = img.permute(1, 2, 0)
                # scale each detection back up to the image
                scale = torch.Tensor([img.shape[1], img.shape[0],
                                      img.shape[1], img.shape[0]])

                recognized_boxes = []
                recognized_scores = []

                # [1,2,200,5]
                # we only care about math class
                # hence select detections[image_id, class, detection_id, detection_score]
                # class=1 for math
                i = 1
                j = 0

                while j < detections.size(2) and detections[k, i, j, 0] >= thres:  # TODO it was 0.6

                    score = detections[k, i, j, 0]
                    pt = (detections[k, i, j, 1:] * self.args.window).cpu().numpy()
                    coords = (pt[0] + x_l, pt[1] + y_l, pt[2] + x_l, pt[3] + y_l)
                    # coords = (pt[0], pt[1], pt[2], pt[3])
                    recognized_boxes.append(coords)
                    recognized_scores.append(score.cpu().numpy())

                    j += 1
                    print(j)

                save_boxes(self.args, recognized_boxes, recognized_scores, img_id)
        self.boxes = recognized_boxes
        self.scores = recognized_scores


    def DetectAny (self, thres, image):
        t = _img_to_tensor(image).unsqueeze(0)
        # fix box coordinates to image pixel coordinates
        self.Detect(thres, t)
        # coor_boxes = FixImgCoordinates(image, self.boxes)
        # new_boxes = self.Voting(t, coor_boxes)
        # self.boxes = coor_boxes

        return self.boxes, self.scores

    def Voting(self, image, math_regions):
        original_width = image.shape[3]
        original_height = image.shape[2]
        thresh_votes = 30

        votes = np.zeros(shape=(original_height, original_width))

        for box in math_regions:
            votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = \
                votes[int(box[1]):int(box[3]), int(box[0]):int(box[2])] + 1

        votes[votes < thresh_votes] = 0
        votes[votes >= thresh_votes] = 1

        im_bw = convert_to_binary(image)

        structure = np.ones((3, 3), dtype=np.int)
        labeled, ncomponents = label(votes, structure)

        boxes = []
        indices = np.indices(votes.shape).T[:, :, [1, 0]]

        for i in range(ncomponents):

            labels = (labeled == (i+1))
            pixels = indices[labels.T]

            if len(pixels) < 1:
                continue

            box = [min(pixels[:, 0]), min(pixels[:, 1]), max(pixels[:, 0]), max(pixels[:, 1])]

            # if args.postprocess:
                # expansion to correctly fit the region
            box = fit_box.adjust_box(im_bw, box)

            # if box has 0 width or height, do not add it in the final detections
            if feature_extractor.width(box) < 1 or feature_extractor.height(box) < 1:
                continue

            boxes.append(box)
        return boxes


def get_img():
    img = cv2.imread('images/3.jpg', cv2.IMREAD_COLOR)
    cimg = img[0:3000, 1000:4000].astype(np.float32)
    return cimg

md = MathDetector('AMATH512_e1GTDB.pth', ArgStub())
# a = get_img()

a = cv2.imread('images/test/1.jpg', cv2.IMREAD_COLOR)
# exit(0)

b, s = md.DetectAny(0.2, a)

md.Voting()

# print(len(s[0]))

DrawAllBoxes([a, ], b)
cv2.imwrite('images/res.png', a)
