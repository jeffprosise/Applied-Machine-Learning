import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class YOLO3:
    width = 416 # Width of input images for predictions
    height = 416 # Height of input images for predictions

    # Labels for the COCO dataset
    labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', \
              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', \
              'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', \
              'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', \
              'ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', \
              'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', \
              'chair', 'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tv monitor', 'laptop', 'mouse', \
              'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def _decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    netout_c = netout.copy() # Preserve original prediction data
    grid_h, grid_w = netout_c.shape[:2]
    nb_box = 3
    netout_r = netout_c.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout_r.shape[-1] - 5

    boxes = []

    netout_r[..., :2]  = _sigmoid(netout_r[..., :2])
    netout_r[..., 4:]  = _sigmoid(netout_r[..., 4:])
    netout_r[..., 5:]  = netout_r[..., 4][..., np.newaxis] * netout_r[..., 5:]
    netout_r[..., 5:] *= netout_r[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout_r[int(row)][int(col)][b][4]
            
            if(objectness.all() <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout_r[int(row)][int(col)][b][:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout_r[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)

    return boxes

def _correct_yolo_boxes(boxes, image_h, image_w):
    for box in boxes:
        box.xmin = int(box.xmin * image_w)
        box.xmax = int(box.xmax * image_w)
        box.ymin = int(box.ymin * image_h)
        box.ymax = int(box.ymax * image_h)
        
def _do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if _bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def decode_predictions(predictions, image_w, image_h, input_w=YOLO3.width, input_h=YOLO3.height, min_score=0.9):
    boxes, output = [], []
    anchors = [[116,90, 156,198, 373,326],  [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

    for i in range(len(predictions)):
        boxes += _decode_netout(predictions[i][0], anchors[i], min_score, input_h, input_w)

    _correct_yolo_boxes(boxes, image_h, image_w)
    _do_nms(boxes, 0.45) # NMS threshold = 0.45

    for box in boxes:
        for i in range(len(YOLO3.labels)):
            if box.classes[i] > min_score:
                box.label = YOLO3.labels[i]
                box.score = box.classes[i]
                output.append(box)

    return output

def draw_boxes(filename, boxes, figsize=(12, 8)):
    image = plt.imread(filename)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'xticks': [], 'yticks': []})
    ax.imshow(image)

    for box in boxes:
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='red', lw=2)
        ax.add_patch(rect)
        label = f'{box.label} ({box.score:.0%})'
        ax.text(x1 + (width / 2), y1, label, color='white', backgroundcolor='red', ha='center', va='bottom', fontweight='bold', bbox=dict(color='red'))