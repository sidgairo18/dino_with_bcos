# Copyright (c) Facebook, Inc. and its affiliates.
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
"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import uuid
from pathlib import Path

import main_dino
import submitit

import torch


def parse_args():
    parser = argparse.ArgumentParser("Submitit for DINO", parents=[main_dino.get_args_parser()])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=716, type=int, help="Duration of the job in minutes")

    #parser.add_argument("--timeout_min", default=None, type=int, help="Job duration in minutes")
    parser.add_argument("--partition", default="gpu16,gpu20,gpu22", type=str, help="Partition where to submit")

    parser.add_argument('--comment', default=None, type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder(shared_folder="/checkpoint/") -> Path:
    user = os.getenv("USER")
    if Path(shared_folder).is_dir():
        #p = Path(f"/checkpoint/{user}/experiments")
        p = Path(shared_folder) / Path(f"{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(output_folder = None):
    # Init file must not exist, but it's parent dir must exist.
    if output_folder is None or output_folder == "":
        os.makedirs(str(get_shared_folder()), exist_ok=True)
        init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    else:
        output_folder = Path(output_folder)
        init_file = output_folder / f"{uuid.uuid4().hex}_init"

    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_dino

        self._setup_gpu_args()
        main_dino.train_dino(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(output_folder=str(self.args.output_dir).replace("%j", '')).as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()

    # hacky way to find lastest checkpoint in args.output_dir
    # then set --init_checkpoint accordingly if --start_checkpoint is None
    if args.init_checkpoint is None and args.start_checkpoint is None:
        last_epoch = [-1, None]
        sub_paths = os.listdir(args.output_dir)
        for spth in sub_paths:
            if os.path.isdir(args.output_dir + '/' + spth):
                ckpt_path = args.output_dir + '/'+ spth + '/' + 'checkpoint.pth'
                if os.path.exists(ckpt_path):
                    curr_ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
                    if last_epoch[0] < curr_ckpt['epoch']:
                        last_epoch[0] = curr_ckpt['epoch']
                        last_epoch[1] = ckpt_path

        
        if last_epoch[0] > -1:
            os.system('cp {} {}'.format(last_epoch[1], args.output_dir + '/' + 'checkpoint_init.pth'))
            args.init_checkpoint = args.output_dir + '/' + 'checkpoint_init.pth'
            print("Init checkpoint found at {}".format(args.init_checkpoint), last_epoch[0])

    if args.output_dir == "":
        args.output_dir = get_shared_folder() / "%j"
    else:
        args.output_dir = Path(args.output_dir) / "%j"
    Path(str(args.output_dir).replace("%j", '')).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=300)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}

    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb= 110 * num_gpus_per_node,
        #mem_gb= 42 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        #cpus_per_task=4,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=2*4*60,
        **kwargs
    )

    executor.update_parameters(name="dino")

    args.dist_url = get_init_file(output_folder=str(args.output_dir).replace("%j", '')).as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
