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
from agents.mdoc_agent import MDocAgent
import hydra
import pdb

@hydra.main(config_path="../config", config_name="base", version_base="1.2")
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.mdoc_agent.cuda_visible_devices
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    for agent_config in cfg.mdoc_agent.agents:
        agent_name = agent_config.agent
        model_name = agent_config.model
        agent_cfg = hydra.compose(config_name="agent/"+agent_name, overrides=[]).agent
        model_cfg = hydra.compose(config_name="model/"+model_name, overrides=[]).model
        agent_config.agent = agent_cfg
        agent_config.model = model_cfg
    cfg.mdoc_agent.sum_agent.agent = hydra.compose(config_name="agent/"+cfg.mdoc_agent.sum_agent.agent, overrides=[]).agent
    cfg.mdoc_agent.sum_agent.model = hydra.compose(config_name="model/"+cfg.mdoc_agent.sum_agent.model, overrides=[]).model

    dataset_type=cfg.dataset_type
    input_type=cfg.input_type
    latex_path=cfg.latex_path

    dataset = BaseDataset(cfg.dataset) 
    mdoc_agent = MDocAgent(cfg.mdoc_agent)

    if input_type=='structured-input':
        ##use structured text+image as input
        mdoc_agent.predict_dataset_structure(dataset, data_type=dataset_type, flag=0,latex_path=latex_path)
    elif input_type=='image':
        ##use image as input 
        mdoc_agent.predict_dataset(dataset, data_type=dataset_type)
    else:
        ##use OCR text+image as input
        mdoc_agent.predict_dataset_general(dataset, data_type=dataset_type)

if __name__ == "__main__":
    main()