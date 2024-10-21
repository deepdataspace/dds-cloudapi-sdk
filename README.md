# dds-cloudapi-sdk

---

<div align="center">
<p align="center">

<!-- prettier-ignore -->
**The Python SDK for the DDS Cloud API.**
---

<!-- prettier-ignore -->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI python](https://img.shields.io/pypi/pyversions/dds-cloudapi-sdk)](https://pypi.org/project/dds-cloudapi-sdk)
[![PyPI version](https://img.shields.io/pypi/v/dds-cloudapi-sdk)](https://pypi.org/project/dds-cloudapi-sdk)
![PyPI - Downloads](https://img.shields.io/pypi/dm/dds-cloudapi-sdk)

</p>
</div>

---

The dds-cloudapi-sdk is a Python package designed to simplify interactions with the DDS Cloud API. It features:

- **Straightforward** APIs
- **Unified** interfaces
- **Handy** utilities

## Installation

You can get the SDK library directly from PyPi:

```shell
pip install dds-cloudapi-sdk
```

## Quick Start

Below is a straightforward example for the popular IVP algorithm:

```python
# 1. Initialize the client with your API token.
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client

token = "Your API Token Here"
config = Config(token)
client = Client(config)

# 2. Upload local image to the server and get the URL.
infer_image_url = "https://dev.deepdataspace.com/static/04_a.ae28c1d6.jpg"
# infer_image_url = client.upload_file("path/to/infer/image.jpg")  # you can also upload local file for processing
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
# task.set_request_timeout(10)  # set the request timeout in seconds，default is 5 seconds

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

```

For more details on using the SDK, please refer to the [DDS CloudAPI SDK Reference](https://cloudapi-sdk.deepdataspace.com)

## 3. Apply for an API Token
Our API is in private beta. Contact us at [deepdataspace_dm@idea.edu.cn](mailto:deepdataspace_dm@idea.edu.cn) to apply for a free API token.  
Please include a brief introduction to your research or project and how you plan to use the API in your application.  
We're dedicated to supporting academic research and education and welcome any questions or suggestions.

## 4. License

This project is released under
the [Apache 2.0 License](https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/LICENSE).

```text
Copyright 2023-present, IDEA

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
