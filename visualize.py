"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

cut_size = 44

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


for i in range(10):
    raw_img = io.imread('images/' + str(i+1) + '.jpg')
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]

    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    net = VGG('VGG19')
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model_pre.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
    net.eval()

    ncrops, c, h, w = np.shape(inputs)
    print(ncrops)
    print(c, h, w)
    inputs = inputs.view(-1, c, h, w)
    inputs = inputs.cuda()
    with torch.no_grad():
        outputs = net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)

    plt.rcParams['figure.figsize'] = (13.5, 5.5)
    axes = plt.subplot(1, 2, 1)
    plt.imshow(raw_img)
    plt.xlabel('Input Image', fontsize=16)
    axes.set_xticks([])
    axes.set_yticks([])
    plt.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

    plt.subplot(1, 2, 2)
    ind = 0.1 + 0.6 * np.arange(len(class_names))  # the x locations for the groups
    width = 0.4  # the width of the bars: can also be len(x) sequence
    color_list = ['red', 'orangered', 'darkorange', 'limegreen', 'darkgreen', 'royalblue', 'navy']
    for j in range(len(class_names)):
        plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    plt.title("Emotion", fontsize=24)
    plt.xlabel("Category", fontsize=20)
    plt.ylabel("Score", fontsize=20)
    plt.xticks(ind, class_names, fontsize=12)

    # plt.show()
    plt.savefig(os.path.join('images/results/' + str(i+1) + '.png'))
    plt.close()

    print("The Expression is %s" % str(class_names[int(predicted.cpu().numpy())]))
