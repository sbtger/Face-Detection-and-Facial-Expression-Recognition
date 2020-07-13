#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
from skimage.transform import resize
import time
from math import ceil
from PIL import Image
import MNN
import cv2
import numpy as np
import matplotlib.pyplot as plt
import box_utils_numpy as box_utils
import transforms as transforms
from models import *

parser = argparse.ArgumentParser(description='run ultraface with MNN in py')
parser.add_argument('--model_path', default="./RFB-320.mnn", type=str, help='model path')
parser.add_argument('--input_size', default="320,240", type=str,
                    help='define network input size,format: width,height')
parser.add_argument('--threshold', default=0.7, type=float, help='score threshold')
parser.add_argument('--imgs_path', default="./imgs", type=str, help='imgs dir')
parser.add_argument('--results_path', default="./results", type=str, help='results dir')
args = parser.parse_args()

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
strides = [8, 16, 32, 64]

cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [ceil(size / stride) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)
    return priors


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    print("priors nums:{}".format(len(priors)))
    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def inference():
    input_size = [int(v.strip()) for v in args.input_size.split(",")]
    priors = define_img_size(input_size)
    result_path = args.results_path
    imgs_path = args.imgs_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    listdir = os.listdir(imgs_path)
    counter = 1
    for file_path in listdir:
        img_path = os.path.join(imgs_path, file_path)
        image_ori = cv2.imread(img_path)
        interpreter = MNN.Interpreter(args.model_path)
        session = interpreter.createSession()
        input_tensor = interpreter.getSessionInput(session)
        image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, tuple(input_size))
        image2 = image
        image = image.astype(float)
        image = (image - image_mean) / image_std
        image = image.transpose((2, 0, 1))
        tmp_input = MNN.Tensor((1, 3, input_size[1], input_size[0]), MNN.Halide_Type_Float, image, 1)
        input_tensor.copyFrom(tmp_input)
        time_time = time.time()
        interpreter.runSession(session)
        scores = interpreter.getSessionOutput(session, "scores").getData()
        boxes = interpreter.getSessionOutput(session, "boxes").getData()
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        # print("inference time: {} s".format(round(time.time() - time_time, 4)))
        boxes = box_utils.convert_locations_to_boxes(boxes, priors, center_variance, size_variance)
        boxes = box_utils.center_form_to_corner_form(boxes)
        boxes, labels, probs = predict(image_ori.shape[1], image_ori.shape[0], scores, boxes, args.threshold)
        # -----------------------------------------------------------------------------------------------------------
        class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        net = VGG('VGG19')
        checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model_pre.t7'))
        net.load_state_dict(checkpoint['net'])
        net.cuda()
        net.eval()
        # -----------------------------------------------------------------------------------------------------------
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            # print(image_ori.shape)
            inputs = image_ori[box[0]:box[2], box[1]:box[3], :]
            if inputs.shape[0] == 0 or inputs.shape[1] == 0:
                break
            # print(inputs.shape)
            gray = rgb2gray(inputs)
            gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
            img = gray[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)
            inputs = transform_test(img)

            ncrops, c, h, w = np.shape(inputs)
            inputs = inputs.view(-1, c, h, w)
            inputs = inputs.cuda()
            with torch.no_grad():
                outputs = net(inputs)
                outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
                score = F.softmax(outputs_avg, dim=0)
                _, predicted = torch.max(outputs_avg.data, 0)
            num_mood = int(predicted.cpu().numpy())
            str1 = "%s %.2f" % (class_names[num_mood], score.data.cpu().numpy()[num_mood])
            cv2.rectangle(image_ori, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(image_ori, str1,
                        (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            print(str1)
        print(file_path)
        cv2.imwrite(os.path.join(result_path, file_path), image_ori)
        # print("result_pic is written to {}".format(os.path.join(result_path, file_path)))
        # ncv2.imshow("UltraFace_mnn_py", image_ori)
        counter += 1
        cv2.waitKey(-1)
    # plt.show()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()
