#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify, render_template
import utils
import model as modellib
import visualize
from config import Config
from model import log
MAIN_DIRECTORY = os.path.abspath("../")
MODEL_DIRECTORY = os.path.join(MAIN_DIRECTORY, "logs")
# In[7]:
english_lst=['pudding/custard','smashed potatoes','carrots','spanich','veal breaded cutlet','oranges','scallops','beans','bread','yogurt','pizza','pasta']


app = Flask(__name__)


# In[9]:


@app.route('/')
def home():
    class FoodConfig(Config):
        NAME='food'
        GPU_COUNT=1
        IMAGES_PER_GPU=1
        NUM_CLASSES = 1 + 12 
        RPN_ANCHOR_SCALES = (4,8,16, 32,64)
        TRAIN_ROIS_PER_IMAGE = 32
        STEPS_PER_EPOCH = 100
        VALIDATION_STEPS = 10
        LEARNING_RATE=0.02
    class InferenceConfig(FoodConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIRECTORY)

    model_path='model_044.h5'
    print(model_path)
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    image = skimage.io.imread("valimage.jpg.")
    results = model.detect([image], verbose=1)
    r = results[0]
    class_names=['BG','pudding/custard','smashed potatoes','carrots','spanich','veal breaded cutlet','oranges','scallops','beans','bread','yogurt','pizza','pasta'];
 
    calorie_per_sq_inch={'smashed potatoes':1.4778,'carrots':0.7256,'spanich':0.4102,'veal breaded cutlet':4.4247,'scallops':0.9823,'beans':0.5486,'pizza':6.2477,'pasta':3.5398}
    calorie_per_unit={'pudding/custard':130,'oranges':45,'bread':130,'yogurt':102}
##    masked_plate_pixels=1130972
    real_plate_size=12
    real_plate_area=113.04
    pixels_per_inch_sq=masked_plate_pixels/real_plate_area
    calories=[]
    items=[]

    for i in range(r['masks'].shape[-1]):
      masked_food_pixels=r['masks'][:,:,i].sum()
      class_name=class_names[r['class_ids'][i]]
      real_food_area=masked_food_pixels/pixels_per_inch_sq
      if class_name in calorie_per_unit:
        calorie=calorie_per_unit[class_name]
      else:
        calorie=calorie_per_sq_inch[class_name]*real_food_area
      calories.append(calorie)
      items.append(class_name)
      return {"class_name":class_name,"calorie":format(int(calorie))}
    
if __name__ == "__main__":
    app.run(debug=True)
