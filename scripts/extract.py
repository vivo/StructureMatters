# -----------------------------------------------------------------------------
# Copyright 2025 vivo Mobile Communication Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mydatasets.base_dataset import BaseDataset
import hydra

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    dataset = BaseDataset(cfg.dataset)
    dataset.extract_content()

if __name__ == "__main__":
    main()