dds-cloudapi-sdk
============================================

dds-cloudapi-sdk is a python package simplifying the user interactions with the DDS Cloud API with these features:

- **Straightforward** APIs
- **Unified** interfaces
- **Handy** utilities

Installation
------------
You can get the SDK library directly from PyPi::

    pip install dds-cloudapi-sdk

Quick Start
-----------
Here is a simple example of the popular IVP algorithm::

    # 1. Initialize the client with your API token.
    from dds_cloudapi_sdk import Config
    from dds_cloudapi_sdk import Client

    token = "Your API Token Here"
    config = Config(token)
    client = Client(config)

    # 2. Optional: Upload local image to the server and get the URL.
    infer_image_url = client.upload_file("path/to/infer/image.jpg")
    prompt_image_url = infer_image_url  # use the same image for prompt

    # 3. Create a task with proper parameters.
    from dds_cloudapi_sdk.tasks import IVPTask
    from dds_cloudapi_sdk.tasks import RectPrompt
    from dds_cloudapi_sdk.tasks import LabelTypes

    task = IVPTask(
        prompt_image_url=prompt_image_url,
        prompts=[RectPrompt(rect=[475.18, 550.20, 548.10, 599.92], is_positive=True)],
        infer_image_url=infer_image_url,
        infer_label_types=[LabelTypes.BBox, LabelTypes.Mask],  # infer both bbox and mask
    )

    # 4. Run the task and get the result.
    client.run_task(task)

    # 5. Parse the result.
    from dds_cloudapi_sdk.tasks.ivp import TaskResult
    result: TaskResult = task.result

    mask_url = result.mask_url  # the image url with all masks drawn on
    objects = result.objects  # the list of detected objects
    for idx, obj in enumerate(objects):
        # get the detection score
        print(obj.score)  # 0.42

        # get the detection bbox
        print(obj.bbox)  # [635.0, 458.0, 704.0, 508.0]

        # get the detection mask, it's of RLE format
        print(obj.mask.counts)  # ]o`f08fa14M3L2O2M2O1O1O1O1N2O1N2O1N2N3M2O3L3M3N2M2N3N1N2O...

        # convert the RLE format to RGBA image
        mask_image = task.rle2rgba(obj.mask)
        print(mask_image.size)  # (1600, 1170)

        # save the image to file
        mask_image.save(f"data/mask_{idx}.png")

Documentation
-------------
As the example shown above, there are three major steps to run a algorithm:

- **1. Config**: Initialize a config object with the API token.
- **2. Client**: Initialize a client object with the config above.
- **3. Tasks**: Trigger a task with the client above and get the result.

The first two steps are exactly the same for all algorithms.

So the rest of this doc will introduce the Config and Client briefly and then dive into the details of different usage patterns of every algorithms(task classes).

.. toctree::
   :maxdepth: 2

   dds_cloudapi_sdk/config
   dds_cloudapi_sdk/client

.. toctree::
   :maxdepth: 3

   dds_cloudapi_sdk/tasks
