import requests
import base64
import numpy as np 
import json 

from deepface.commons.functions import detectFace, findThreshold
from deepface.commons.distance import findCosineDistance


URL = "https://facenet-deepface-model-woomurf.endpoint.ainize.ai/v1/models/facenet:predict"

def image_open(image_path):
    image = open(image_path, "rb")
    image = base64.b64encode(image.read()).decode("utf-8")

    image = 'data:image/png;base64,' + image

    return image

def get_representation(image):
    data = {
        "instances": image
    }
    data = json.dumps(data)
    response = requests.post(URL, data=data)
    return response

def verifyTest():
    image1 = image_open("./sample1.png")
    image2 = image_open("./sample2.png")

    input_shape_y = 160
    input_shape_x = 160

    img1 = detectFace(image1,(input_shape_y, input_shape_x))
    img2 = detectFace(image2,(input_shape_y, input_shape_x))
    print("detect face")

    img1 = img1.tolist()
    img2 = img2.tolist()

    try:
        img1_representation = get_representation(img1)
        img2_representation = get_representation(img2)
        print("Get representation")
    except Exception as e:
        print(e)
        print("Fail run model.")
        return

    img1_result = img1_representation.text
    img2_result = img2_representation.text

    img1_result = json.loads(img1_result)
    img2_result = json.loads(img2_result)


    img1_result = img1_result['predictions']
    img2_result = img2_result['predictions']

    img1_result = np.array(img1_result)[0,:]
    img2_result = np.array(img2_result)[0,:]

    distance = findCosineDistance(img1_result, img2_result)
    threshold = findThreshold('Facenet', 'cosine')

    if distance <= threshold:
        result = True
    else:
        result = False
    
    result_obj = {
        "distance": distance,
        "threshold": threshold,
        "verified": result
    }
    return result_obj

print(verifyTest())