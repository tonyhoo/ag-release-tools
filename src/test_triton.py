from tritonclient.utils import *
import tritonclient.http as httpclient
import sys
import subprocess
import numpy as np
import pandas as pd
import base64
import time

def start_triton_server():
    triton_path = '/opt/tritonserver/bin/tritonserver'  # Adjust this path as needed
    model_repository = '/opt/ml/model'  # Adjust this path as needed
    print("Starting the triton server ...")

    # Define TritonServer startup command
    triton_startup_cmd = [
        triton_path,
        '--model-repository=' + model_repository,
        # Additional flags can be added here
        '--log-verbose=1',
        '--log-error=1',
        '--log-info=1',
    ]

    # Run TritonServer in a subprocess
    triton_server_process = subprocess.Popen(triton_startup_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(10)
    return triton_server_process

def test_triton():
    triton_server_process = start_triton_server()
    start_inference()
    triton_server_process.terminate()
    triton_server_process.wait()

    

def start_inference():
    print("Starting inference ...")
    model_name = "autogluon"
    shape = [1]
    test_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/test.csv")
    df_parquet = test_data.to_parquet()
    inputs = [
        httpclient.InferInput("INPUT0", shape, "BYTES"),
    ]

    inputs[0].set_data_from_numpy(
        np.array([base64.b64encode(df_parquet)]), binary_data=True
    )

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
    ]

    with httpclient.InferenceServerClient("localhost:8000") as client:
        response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

        result = response.get_response()
        output0_data = response.as_numpy("OUTPUT0")

        print(f"Get server resposne {output0_data}")
        print(f"PASS: {model_name}")
