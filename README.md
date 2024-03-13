# dds-cloudapi-sdk


---

<div align="center">
<p align="center">

<!-- prettier-ignore -->
**The Python SDK for calling the DDS Cloud API.**
---

<!-- prettier-ignore -->

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyPI python](https://img.shields.io/pypi/pyversions/dds-cloudapi-sdk)](https://pypi.org/project/dds-cloudapi-sdk)
[![PyPI version](https://img.shields.io/pypi/v/dds-cloudapi-sdk)](https://pypi.org/project/dds-cloudapi-sdk)
![PyPI - Downloads](https://img.shields.io/pypi/dm/dds-cloudapi-sdk)

</p>
</div>

---

## 1. Installation
```bash
pip install -U dds-cloudapi-sdk
```

## 2. How to Use
The following is a simple example of how to use the SDK to call the DDS Cloud API for interactive visual prompt (IVP) tasks.

```python
# 1. Initialize the client with your API token.
from dds_cloudapi_sdk import Client

token = "Your API Token Here"
client = Client(token)

# 2. Optional: Upload local image to the server and get the URL.
infer_image_url = client.upload_file("data/test_ivp.jpg")
prompt_image_url = client.upload_file("data/test_ivp.jpg")

# 3. Create a task with proper parameters.
from dds_cloudapi_sdk.tasks import IVPTask
from dds_cloudapi_sdk.tasks import RectPrompt
from dds_cloudapi_sdk.tasks import LabelTypes

task = IVPTask(
    prompt_image_url=prompt_image_url,
    prompts=[
        RectPrompt(rect=[475.18413597733706, 550.1983002832861, 548.1019830028329, 599.915014164306], is_positive=True)
    ],
    infer_image_url=infer_image_url,
    infer_label_types=[LabelTypes.BBox, LabelTypes.Mask],
)

# 4. Run the task and get the result.
client.run_task(task)

# 5. Parse the result.
from dds_cloudapi_sdk.tasks.ivp import TaskResult

print(task.status)

result: TaskResult = task.result
print(task.result)

mask_url = result.mask_url  # the url with all masks drawn on
objects = result.objects  # the list of detected objects
for idx, obj in enumerate(objects):
    # get the detection score
    print(obj.score)  # 0.42

    # get the detection bbox
    print(obj.bbox)  # [635.0, 458.0, 704.0, 508.0]

    # get the detection mask, it's of RLE format
    print(obj.mask.counts)  # ]o`f08fa14M3L2O2M2O1O1O1O1N2O1N2O1N2N3M2O3L3M3N2M2N3N1N2O...

    # convert the RLE format to RGBA image
    mask_image = task.rle2rgba(obj.mask.counts)
    print(mask_image.size)  # (1600, 1170)

    # save the image to file
    mask_image.save(f"data/mask_{idx}.png")
```
Please visit the API documentation for more details on how to use the SDK: [DDS CloudAPI SDK Reference](https://dds-cloudapi-sdk-docs.deepdataspace.com)


## 3. Apply for an API Token
Our API is currently in private beta. Please contact us at [Wei Liu, weiliu@idea.edu.cn](mailto:weiliu@idea.edu.cn) to apply for an API token.  
We are fully committed to support academic research and education, please feel free to reach out to us for any questions or suggestions.

## 4. License
This project is released under the [Apache 2.0 License](https://github.com/deepdataspace/dds-cloudapi-sdk/blob/main/LICENSE).
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
