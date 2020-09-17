import requests
import base64
import numpy as np 
import json 

from deepface.commons.functions import detectFace, findThreshold
from deepface.commons.distance import findCosineDistance
from deepface.extendedmodels import Age

URL = "https://emotion-deepface-model-woomurf.endpoint.ainize.ai/v1/models/emotion:predict"

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

def analyzeTest():
    image = image_open("./sample1.png")
    if "emotion" in URL:
        img = detectFace(image, (48,48), grayscale=True)
    else:
        img = detectFace(image,(224,224))
    img = img.tolist()

    img_representation = get_representation(img)
    img_representation = img_representation.text
    img_representation = json.loads(img_representation)['predictions']

    img_result = np.array(img_representation)[0,:]

    if "emotion" in URL:
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

    elif "gender" in URL:
        if np.argmax(img_result) == 0:
            gender = "woman"
        else:
            gender = "man"
        
        return gender
    
    elif "race" in URL:
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
    elif "age" in URL:
        age = Age.findApparentAge(img_result)
        return age


print(analyzeTest())