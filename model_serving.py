from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
from deepface.extendedmodels import Age, Gender, Race, Emotion
from deepface.commons.functions import detectFace
import tensorflow as tf 
import numpy as np 
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def

def model_serve(model_name):
    
    tf.reset_default_graph()
    with tf.Session() as sess : 
        # sess.run(tf.global_variables_initializer())

        if model_name == "VGGFace":
            model = VGGFace.loadModel()
        elif model_name == "OpenFace":
            model = OpenFace.loadModel()
        elif model_name == "Facenet":
            model = Facenet.loadModel()
        elif model_name == "FbDeepFace":
            model = FbDeepFace.loadModel()
        elif model_name == "DeepID":
            model = DeepID.loadModel() 
        elif model_name == "Age":
            model = Age.loadModel()
        elif model_name == "Emotion":
            model = Emotion.loadModel()
        elif model_name == "Gender":
            model = Gender.loadModel()
        elif model_name == "Race":
            model = Race.loadModel()

        folder = "./" + model_name.lower() + "/1"

        saver = tf.saved_model.builder.SavedModelBuilder(folder)

        signature = predict_signature_def(
            inputs = {"instances": model.inputs[0]},
            outputs = {"output": model.outputs[0]},
        )

        saver.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
        )

        saver.save()