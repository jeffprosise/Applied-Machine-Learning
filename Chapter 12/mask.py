class COCO:
    classes = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', \
              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', \
              'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', \
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', \
              'ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', \
              'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', \
              'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', \
              'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

import math
import numpy as np
from PIL import Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cv2

def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)
    
    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible by 32
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

def annotate_image(image, boxes, labels, scores, masks, min_confidence=0.7,
                    mask_color=(255, 0, 255), mask_opacity=0.4, show_boxes=True, show_masks=True):
    # Resize boxes
    ratio = 800.0 / min(image.size[0], image.size[1])
    boxes /= ratio

    fig, ax = plt.subplots(1, figsize=(12,9), subplot_kw={'xticks': [], 'yticks': []})

    image = np.array(image)

    for mask, box, label, score in zip(masks, boxes, labels, scores):
        if score < min_confidence:
            continue

        if (show_masks):
            mask = mask[0, :, :, None]
            int_box = [int(i) for i in box]
            mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
            mask = mask > 0.5
            im_mask = np.zeros(image.shape[:-1], dtype=np.uint8)
            x_0 = max(int_box[0], 0)
            x_1 = min(int_box[2] + 1, image.shape[1])
            y_0 = max(int_box[1], 0)
            y_1 = min(int_box[3] + 1, image.shape[0])
            mask_y_0 = max(y_0 - box[1], 0)
            mask_y_1 = mask_y_0 + y_1 - y_0
            mask_x_0 = max(x_0 - box[0], 0)
            mask_x_1 = mask_x_0 + x_1 - x_0
            im_mask[y_0:y_1, x_0:x_1] = mask[mask_y_0 : mask_y_1, mask_x_0 : mask_x_1]
            im_mask = im_mask[:, :, None]

            contours, hierarchy = cv2.findContours(im_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            im_copy= image.copy()
            
            for i, cont in enumerate(contours):
                im_copy = cv2.drawContours(im_copy, [cont], -1, mask_color, thickness=cv2.FILLED)
                image = cv2.addWeighted(im_copy, mask_opacity, image, 1.0 - mask_opacity, 0.0)

        if (show_boxes):
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = Rectangle((x, y), w, h, edgecolor='r', lw=2, facecolor='none')
            text = f'{COCO.classes[label]} ({score:.1%})'
            ax.text(x + (w / 2), y, text, color='white', backgroundcolor='red',
                    ha='center', va='bottom', fontweight='bold', bbox=dict(color='red'))    
            ax.add_patch(rect)
        
    ax.imshow(image)

def change_background(session, foregroundImage, backgroundImage, min_confidence=0.7):
    #Submit foreground image to Mask R-CNN
    img_data = preprocess(foregroundImage)
    input_name = session.get_inputs()[0].name
    result = session.run(None, { input_name: img_data })

    # Extract the results
    boxes = result[0]
    labels = result[1]
    scores = result[2]
    masks = result[3]

    # Scale boxes to match original image size
    ratio = 800.0 / min(foregroundImage.size[0], foregroundImage.size[1])
    boxes /= ratio
    
    foregroundImage = np.array(foregroundImage)
    backgroundImage = np.array(backgroundImage)
    backgroundImage = cv2.resize(backgroundImage, foregroundImage.shape[1::-1])
    
    fig, ax = plt.subplots(1, figsize=(12,9), subplot_kw={'xticks': [], 'yticks': []})
    
    # For each segmentation mask, copy the pixels inside the mask
    # from the foreground image to the background image
    for mask, box, label, score in zip(masks, boxes, labels, scores):
        if score <= min_confidence:
            continue

        mask = mask[0, :, :, None]
        int_box = [int(i) for i in box]
        mask = cv2.resize(mask, (int_box[2]-int_box[0]+1, int_box[3]-int_box[1]+1))
        mask = mask > 0.5
        im_mask = np.zeros(foregroundImage.shape[:-1], dtype=np.uint8)
        x_0 = max(int_box[0], 0)
        x_1 = min(int_box[2] + 1, foregroundImage.shape[1])
        y_0 = max(int_box[1], 0)
        y_1 = min(int_box[3] + 1, foregroundImage.shape[0])
        mask_y_0 = max(y_0 - box[1], 0)
        mask_y_1 = mask_y_0 + y_1 - y_0
        mask_x_0 = max(x_0 - box[0], 0)
        mask_x_1 = mask_x_0 + x_1 - x_0
        im_mask[y_0:y_1, x_0:x_1] = mask[mask_y_0 : mask_y_1, mask_x_0 : mask_x_1]
        im_mask = im_mask[:, :, None]

        contours, hierarchy = cv2.findContours(im_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create a mask (stencil) of 1s and 0s with 1s denoting pixels that fall
        # inside the contours
        mask_val = 1
        stencil  = np.zeros(foregroundImage.shape[:-1]).astype(np.uint8)
        cv2.fillPoly(stencil, contours, mask_val)

        # Copy pixels in the foreground image that correspond to 1s in the
        # stencil to the background image
        w = stencil.shape[1]
        h = stencil.shape[0]

        for x in range(w):
            for y in range(h):
                if (stencil[y, x] == mask_val):
                    backgroundImage[y, x] = foregroundImage[y, x]

    ax.imshow(backgroundImage)