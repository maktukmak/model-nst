
from google.cloud import aiplatform
from typing import Dict, List, Union
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import base64
import time


def convert_b64(input_file_name):
    """Open image and convert it to Base64"""
    with open(input_file_name, "rb") as input_file:
        jpeg_bytes = base64.b64encode(input_file.read()).decode("utf-8")
    return jpeg_bytes

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = instances if isinstance(instances, list) else [instances]
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    start = time.time()
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print('Inference time:', time.time()-start)
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions

    return predictions