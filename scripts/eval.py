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
from agents.base_agent import Agent
import hydra
import pdb

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    pdb.set_trace()
    cfg.eval_agent.agent = hydra.compose(config_name="agent/"+cfg.eval_agent.agent, overrides=[]).agent
    cfg.eval_agent.model = hydra.compose(config_name="model/"+cfg.eval_agent.model, overrides=[]).model
    dataset = BaseDataset(cfg.dataset)
    eval_agent = Agent(cfg.eval_agent)
    eval_agent.eval_dataset(dataset)
    
if __name__ == "__main__":
    main()