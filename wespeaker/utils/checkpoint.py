# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#               2021 Hongji Wang (jijijiang77@gmail.com)
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

import torch
import logging


def load_checkpoint(model: torch.nn.Module, path: str):
    loaded_state = torch.load(path, map_location='cpu')
    self_state = model.state_dict();
    for name, param in loaded_state.items():
        origname = name;
        if name not in self_state:
            name = name.replace("speaker_extractor.", "");

            if name not in self_state:
                print("%s is not in the model."%origname);
                continue;

        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
            continue;

        self_state[name].copy_(param);
    # checkpoint = torch.load(path, map_location='cpu')
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
    #                                                       strict=False)
    # for key in missing_keys:
    #     logging.warning('missing tensor: {}'.format(key))
    # for key in unexpected_keys:
    #     logging.warning('unexpected tensor: {}'.format(key))




def save_checkpoint(model: torch.nn.Module, path: str):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
