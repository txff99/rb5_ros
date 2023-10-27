#!/usr/bin/env python3

import torch
import cv2
import time

image = cv2.imread('/root/rb5_ws/src/rb5_ros/hw2/imgs/0.png')[..., ::-1]
image = cv2.resize(image, (640, 640))

model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model.eval()

# start_time = time.time()
# for i in range(10):
results = model(image, size=640)
# end_time = time.time()
# duration = end_time - start_time
# print("avg:", duration/10)

results.print()
results.show()