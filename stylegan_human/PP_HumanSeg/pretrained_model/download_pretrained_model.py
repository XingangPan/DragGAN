# coding: utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os

LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(LOCAL_PATH, "../../../", "test")
sys.path.append(TEST_PATH)

from paddleseg.utils.download import download_file_and_uncompress

model_urls = {
    "pphumanseg_lite_portrait_398x224":
    "https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224.tar.gz",
    "deeplabv3p_resnet50_os8_humanseg_512x512_100k":
    "https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/deeplabv3p_resnet50_os8_humanseg_512x512_100k.zip",
    "fcn_hrnetw18_small_v1_humanseg_192x192":
    "https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/fcn_hrnetw18_small_v1_humanseg_192x192.zip",
    "pphumanseg_lite_generic_human_192x192":
    "https://paddleseg.bj.bcebos.com/dygraph/humanseg/train/pphumanseg_lite_generic_192x192.zip",
}

if __name__ == "__main__":
    for model_name, url in model_urls.items():
        download_file_and_uncompress(
            url=url,
            savepath=LOCAL_PATH,
            extrapath=LOCAL_PATH,
            extraname=model_name)

    print("Pretrained model download success!")
