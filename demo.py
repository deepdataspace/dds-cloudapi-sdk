# 1. Initialize the client with your API token.
from dds_cloudapi_sdk.tasks.v2_task import create_task_with_local_image_auto_resize
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk.visualization_util import visualize_result

token = "Your API Token Here"
config = Config(token)
client = Client(config)

# 2. Create a task with proper parameters.
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

# 3. Run the task.
# task.set_request_timeout(10)  # set the request timeout in seconds，default is 5 seconds
client.run_task(task)

# 4. Get the result.
print(task.result)

# 5. Visualize the result to local image.
visualize_result(image_path="path/to/infer/image.jpg", result=task.result, output_dir="path/to/output_dir")
