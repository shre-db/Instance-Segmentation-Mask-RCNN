import cv2 as cv
import numpy as np

# Load Mask RCNN
net = cv.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                   "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

# Generate random colors
colors = np.random.randint(0, 255, (80, 3))

print(colors)

# Load image
img = cv.imread("images/road.jpg")
height, width, _ = img.shape

# Create black image
black_image = np.zeros((height, width, 3), np.uint8)
black_image[:] = (100, 100, 0)

# Detect objects
blob = cv.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
detection_count = boxes.shape[2]

for i in range(detection_count):

    box = boxes[0, 0, i]
    class_id = box[1]
    score = box[2]
    if score < 0.5:
        continue

    # Get box coordinates
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)

    roi = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape

    # Get the mask
    mask = masks[i, int(class_id)]
    mask = cv.resize(mask, (roi_width, roi_height))
    _, mask = cv.threshold(mask, 0.5, 255, cv.THRESH_BINARY)

    cv.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 3)

    # Get mask coordinates
    contours, _ = cv.findContours(np.array(mask, np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    color = colors[int(class_id)]
    for cnt in contours:
        cv.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))

    # cv.imshow("roi", roi)
    # cv.waitKey(0)

cv.imshow("Image", img)
cv.imshow("Black image", black_image)
cv.waitKey(0)
