# 1. Initialize the client with your API token.
from dds_cloudapi_sdk.tasks.v2_task import create_task_with_local_image_auto_resize
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.visualization_util import visualize_result

token = "Your API Token Here"
config = Config(token)
client = Client(config)

# 2. Upload local image to the server and get the URL.
# infer_image_url = "https://dds-frontend.oss-accelerate.aliyuncs.com/static_files/playground/grounding_DINO-1.6/02.jpg"
# infer_image_url = client.upload_file("path/to/infer/image.jpg")  # you can also upload local file for processing

# 3. Create a task with proper parameters.

task = create_task_with_local_image_auto_resize(
    api_path="/v2/task/dinox/detection",
    api_body={
        "model": "DINO-X-1.0",
        # "image": infer_image_url, # not needed for local image
        "prompt": {
            "type": "text",
            "text": "wolf.dog.butterfly"
        },
        "targets": ["bbox"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8
    },
    image_path="path/to/infer/image.jpg")
# task.set_request_timeout(10)  # set the request timeout in seconds，default is 5 seconds

# 4. Run the task.
client.run_task(task)

# 5. Get the result.
print(task.result)

# 6. Visualize the result to local image.
visualize_result(image_path="path/to/infer/image.jpg", result=task.result, output_dir="path/to/output_dir")
