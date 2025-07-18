import abc
import os

import ray

from slime.ray.ray_actor import RayActor


class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port):
        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(start_port=20000)

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])

    @abc.abstractmethod
    def init(self, args, role, with_ref=False):
        raise NotImplementedError

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock):
        raise NotImplementedError

    @abc.abstractmethod
    def set_data_buffer(self, data_buffer):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, with_data_fetching=True):
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, rollout_id):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, iteration, with_optimizer=True):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError
