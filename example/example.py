import requests
import base64
import numpy as np 
import json 
import argparse

from deepface.commons.functions import detectFace, findThreshold
from deepface.commons.distance import findCosineDistance
from deepface.extendedmodels import Age



def image_open(image_path):
    image = open(image_path, "rb")
    image = base64.b64encode(image.read()).decode("utf-8")

    image = 'data:image/png;base64,' + image

    return image

def get_representation(image, URL):
    data = {
        "instances": image
    }
    data = json.dumps(data)
    response = requests.post(URL, data=data)
    return response

def analyzeTest(model, image, URL):
    image = image_open(image)

    if model == "Emotion":
        img = detectFace(image, (48,48), grayscale=True)
    else:
        img = detectFace(image,(224,224))
    img = img.tolist()

    img_representation = get_representation(img, URL)
    img_representation = img_representation.text
    img_representation = json.loads(img_representation)['predictions']

    img_result = np.array(img_representation)[0,:]

    if model == "Emotion":
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        sum_of_predictions = img_result.sum()

        emotion_obj = "\"emotion\": {"
        for i in range(0, len(emotion_labels)):
            emotion_label = emotion_labels[i]
            emotion_prediction = 100 * img_result[i] / sum_of_predictions

            if i > 0: emotion_obj += ", "

            emotion_obj += "\"%s\": %s" % (emotion_label, emotion_prediction)

        emotion_obj += "}"

        emotion_obj += ", \"dominant_emotion\": \"%s\"" % (emotion_labels[np.argmax(img_result)])
     
        return emotion_obj

    elif model == "Gender":
        if np.argmax(img_result) == 0:
            gender = "woman"
        else:
            gender = "man"
        
        return gender
    
    elif model == "Race":
        race_predictions = img_result
        race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

        sum_of_predictions = race_predictions.sum()

        race_obj = "\"race\": {"
        for i in range(0, len(race_labels)):
            race_label = race_labels[i]
            race_prediction = 100 * race_predictions[i] / sum_of_predictions

            if i > 0: race_obj += ", "

            race_obj += "\"%s\": %s" % (race_label, race_prediction)

        race_obj += "}"
        race_obj += ", \"dominant_race\": \"%s\"" % (race_labels[np.argmax(race_predictions)])

        return race_obj
    elif model == "Age":
        age = Age.findApparentAge(img_result)
        return age

def verifyTest(model, image1, image2, URL):
    image1 = image_open(image1)
    image2 = image_open(image2)

    if model == "VGGFace":
        input_shape_y = 224
        input_shape_x = 224
    elif model == "OpenFace":
        input_shape_y = 96
        input_shape_x = 96
    elif model == "FbDeepFace":
        input_shape_y = 152
        input_shape_x = 152
    elif model == "DeepID":
        input_shape_y = 47
        input_shape_x = 55
    elif model == "Facenet":
        input_shape_y = 160
        input_shape_x = 160

    img1 = detectFace(image1,(input_shape_y, input_shape_x))
    img2 = detectFace(image2,(input_shape_y, input_shape_x))
    print("detect face")

    img1 = img1.tolist()
    img2 = img2.tolist()

    try:
        img1_representation = get_representation(img1, URL)
        img2_representation = get_representation(img2, URL)
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

    if model == "VGGFace":
        threshold = findThreshold('VGG-Face', 'cosine')
    elif model == "FbDeepFace":
        threshold = findThreshold('DeepFace', 'cosine')
    else:
        threshold = findThreshold(model, 'cosine')

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example for DeepFace model running")
    parser.add_argument("--mode", default="verify", type=str, help="verify or analyze")
    parser.add_argument("--model", default="VGGFace", type=str, help="models = [VGGFace, OpenFace, Facenet, FbDeepFace, DeepID, Age, Gender, Race, Emotion]")
    parser.add_argument("--image1", default="./sample1.png", type=str, help="image for verify or analze")
    parser.add_argument("--image2", default="./sample2.png", type=str, help="second image for verify")

    args = parser.parse_args()
    mode = args.mode 
    model = args.model

    URL = "https://{}-deepface-model-woomurf.endpoint.ainize.ai/v1/models/{}:predict".format(model.lower(),model.lower())
    if mode == "verify":
        print(verifyTest(model, args.image1, args.image2, URL))
    elif mode == "analyze":
        print(analyzeTest(model, args.image1, URL))
    else:
        print("Check mode.")
