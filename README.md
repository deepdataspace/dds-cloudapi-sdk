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

Below is a straightforward example for the popular DINO-X - Detection algorithm:

```python
# 1. Initialize the client with your API token.
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client

token = "Your API Token Here"
config = Config(token)
client = Client(config)

# 2. Upload local image to the server and get the URL.
infer_image_url = "https://dds-frontend.oss-accelerate.aliyuncs.com/static_files/playground/grounding_DINO-1.6/02.jpg"
# infer_image_url = client.upload_file("path/to/infer/image.jpg")  # you can also upload local file for processing

# 3. Create a task with proper parameters.
from dds_cloudapi_sdk.tasks.v2_task import V2Task

task = V2Task(api_path="/v2/task/dinox/detection", api_body={
    "model": "DINO-X-1.0",
    "image": infer_image_url,
    "prompt": {
        "type":"text",
        "text":"wolf.dog.butterfly"
    },
    "targets": ["bbox"],
    "bbox_threshold": 0.25,
    "iou_threshold": 0.8
})
# task.set_request_timeout(10)  # set the request timeout in seconds，default is 5 seconds

# 4. Run the task.
client.run_task(task)

# 5. Get the result.
print(task.result)

```

## 3. Apply for an API Token
Step 1: [Apply for API Quota](https://cloud.deepdataspace.com/apply-token?from=sdk).  
Step 2: Create projects and get your API token from [here](https://cloud.deepdataspace.com/dashboard/token-key).

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
