# Copyright 2024 EPFL and Apple Inc.
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
def setup_run_name(args):
    if args.run_name == 'auto':
        # This returns the config name after removing the first two parent dirs and extension
        args.run_name = args.config_path.partition('cfgs/')[2].partition('/')[2].replace(".yaml", "")

    if "wandb_run_name" in args and args.wandb_run_name == 'auto':
        # Wandb omits the current parent dir (pretrain, finetune, etc...) as it is part of the wandb project
        args.wandb_run_name = args.run_name.partition('/')[2]

    if "output_dir" in args and 'auto' in args.output_dir:
        args.output_dir = args.output_dir.replace('auto', args.run_name)

    if "s3_save_dir" in args and 'auto' in args.s3_save_dir:
        args.s3_save_dir = args.s3_save_dir.replace('auto', args.run_name)