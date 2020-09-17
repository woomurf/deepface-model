# deepface-model

This repository has a Dockerfile for deploying [DeepFace](https://github.com/serengil/deepface) Models in [Ainize](https://ainize.ai).

## Example 

requirements

- python3
- deepface==0.0.33 (only for example)

```
cd example 

python3 example.py --mode verify --model VGGFace --image1 ./sample1.png --image2 ./sample2.png 
```

mode = [verify, analyze]

model = [VGGFace, OpenFace, FbDeepFace, DeepID, Facenet, Age, Gender, Race, Emotion]

or each branch has a example code. 

```
# in vggface branch

cd example 

python3 example.py 
```

### VGGFace

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=vggface)

#### Usage 

You must use detected face image.

```
# input_shape = (None, 224, 224, 3)
# output_shape = (None, None)
# You can check metadata in https://vggface-deepface-model-woomurf.endpoint.ainize.ai/v1/models/vggface/metadata

URL = "https://vggface-deepface-model-woomurf.endpoint.ainize.ai/v1/models/vggface:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```


### OpenFace

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=openface)

#### Usage 

You must use detected face image.

```
# input_shape = (None, 96, 96, 3)
# output_shape = (None, 128)
# You can check metadata in https://openface-deepface-model-woomurf.endpoint.ainize.ai/v1/models/openface/metadata

URL = "https://openface-deepface-model-woomurf.endpoint.ainize.ai/v1/models/openface:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```

### FbDeepFace 

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=fbdeepface)
#### Usage 

You must use detected face image.

```
# input_shape = (None, 152, 152, 3)
# output_shape = (None, 4096)
# You can check metadata in https://fbdeepface-deepface-model-woomurf.endpoint.ainize.ai/v1/models/fbdeepface/metadata

URL = "https://fbdeepface-deepface-model-woomurf.endpoint.ainize.ai/v1/models/fbdeepface:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```

### DeepID

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=deepid)

#### Usage 

You must use detected face image.

```
# input_shape = (None, 55, 47, 3)
# output_shape = (None, 160)
# You can check metadata in https://deepid-deepface-model-woomurf.endpoint.ainize.ai/v1/models/deepid/metadata

URL = "https://deepid-deepface-model-woomurf.endpoint.ainize.ai/v1/models/deepid:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```

### Facenet

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=facenet)

#### Usage 

You must use detected face image.

```
# input_shape = (None, 160, 160, 3)
# output_shape = (None, 128)
# You can check metadata in https://facenet-deepface-model-woomurf.endpoint.ainize.ai/v1/models/facenet/metadata

URL = "https://facenet-deepface-model-woomurf.endpoint.ainize.ai/v1/models/facenet:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```

### Age 

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=age)

#### Usage 

You must use detected face image.

```
# input_shape = (None, 224, 224, 3)
# output_shape = (None, None)
# You can check metadata in https://age-deepface-model-woomurf.endpoint.ainize.ai/v1/models/age/metadata

URL = "https://age-deepface-model-woomurf.endpoint.ainize.ai/v1/models/age:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```

### Gender

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=gender)

#### Usage 

You must use detected face image.

```
# input_shape = (None, 224, 224, 3)
# output_shape = (None, None)
# You can check metadata in https://gender-deepface-model-woomurf.endpoint.ainize.ai/v1/models/gender/metadata

URL = "https://gender-deepface-model-woomurf.endpoint.ainize.ai/v1/models/gender:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```

### Race 

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=race)

#### Usage 

You must use detected face image.

```
# input_shape = (None, 224, 224, 3)
# output_shape = (None, None)
# You can check metadata in https://race-deepface-model-woomurf.endpoint.ainize.ai/v1/models/race/metadata

URL = "https://race-deepface-model-woomurf.endpoint.ainize.ai/v1/models/race:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```

### Emotion 

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.ai/woomurf/deepface-model?branch=emotion)

#### Usage 

You must use detected face image.

```
# input_shape = (None, 48, 48, 1)
# output_shape = (None, 7)
# You can check metadata in https://emotion-deepface-model-woomurf.endpoint.ainize.ai/v1/models/emotion/metadata

URL = "https://emotion-deepface-model-woomurf.endpoint.ainize.ai/v1/models/emotion:predict"

data = {
    "instances": image 
}

data = json.dumps(data)

response = requests.post(URL, data=data)

response_text = response.text 
result = json.loads(response_text)
result = result['predictions']

```
