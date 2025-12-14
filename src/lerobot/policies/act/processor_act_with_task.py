import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.processor.core import TransitionKey
from lerobot.processor.pipeline import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_STATE


@ProcessorStepRegistry.register(name="act_add_task_embeddings")
class ACTAddTaskEmbeddingsStep(ProcessorStep):
    def __init__(
        self,
        task_embeddings_path: str | Path,
        task_key: str = "task_index",
        embedding_key: str = "task_embedding",
    ):

        super().__init__()
        self.task_embeddings_path = Path(task_embeddings_path)
        self.task_key = task_key
        self.embedding_key = embedding_key
        
        print(f"Loading task embeddings from {self.task_embeddings_path}")
        with open(self.task_embeddings_path, "rb") as f:
            data = pickle.load(f)
            self.task_embeddings = data["task_embeddings"]
            self.embedding_dim = data["embedding_dim"]
            print(f"Loaded {len(self.task_embeddings)} task embeddings (dim={self.embedding_dim})")
        
        # Create reverse mapping from task_name to task_index for evaluation
        self.task_name_to_index = {}
        for task_idx, task_data in self.task_embeddings.items():
            if isinstance(task_data, dict) and "task_name" in task_data:
                self.task_name_to_index[task_data["task_name"]] = task_idx
            elif isinstance(task_data, str):
                # Handle case where task_data is just the task name string
                self.task_name_to_index[task_data] = task_idx
    
    def __call__(self, batch: dict[str, Any]) -> dict[str, Any]:

        complementary_data = batch.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        
        # First try to get task_index
        if self.task_key in complementary_data:
            task_indices = complementary_data[self.task_key]
        elif self.task_key in batch:
            task_indices = batch[self.task_key]
        # If task_index not found, try to get "task" (string) and convert it
        elif "task" in complementary_data:
            task_strings = complementary_data["task"]
            # Handle both list and single string cases
            if isinstance(task_strings, str):
                task_strings = [task_strings]
            # Convert task strings to task indices
            task_indices = []
            for task_str in task_strings:
                if task_str in self.task_name_to_index:
                    task_indices.append(self.task_name_to_index[task_str])
                else:
                    raise ValueError(
                        f"Task '{task_str}' not found in task embeddings. "
                        f"Available tasks: {list(self.task_name_to_index.keys())}"
                    )
            # Convert to numpy array for consistency
            task_indices = np.array(task_indices)
        elif "task" in batch:
            task_strings = batch["task"]
            # Handle both list and single string cases
            if isinstance(task_strings, str):
                task_strings = [task_strings]
            # Convert task strings to task indices
            task_indices = []
            for task_str in task_strings:
                if task_str in self.task_name_to_index:
                    task_indices.append(self.task_name_to_index[task_str])
                else:
                    raise ValueError(
                        f"Task '{task_str}' not found in task embeddings. "
                        f"Available tasks: {list(self.task_name_to_index.keys())}"
                    )
            # Convert to numpy array for consistency
            task_indices = np.array(task_indices)
        else:
            raise ValueError(
                f"Task key '{self.task_key}' or 'task' not found. "
                f"Available batch keys: {list(batch.keys())}, "
                f"Available complementary_data keys: {list(complementary_data.keys())}"
            )
        
        if isinstance(task_indices, torch.Tensor):
            if task_indices.dim() == 0:
                task_indices = task_indices.unsqueeze(0)
            task_indices = task_indices.cpu().numpy()
        
        embeddings = []
        for task_idx in task_indices:
            task_idx = int(task_idx)
            if task_idx not in self.task_embeddings:
                raise ValueError(f"Task index {task_idx} not found in embeddings. Available: {list(self.task_embeddings.keys())}")
            embeddings.append(self.task_embeddings[task_idx]["embedding"])
        
        task_embeddings = torch.stack(embeddings)
        
        observation = batch.get(TransitionKey.OBSERVATION, {})
        if isinstance(observation, dict) and OBS_STATE in observation:
            obs_tensor = observation[OBS_STATE]
            task_embeddings = task_embeddings.to(device=obs_tensor.device, dtype=obs_tensor.dtype)
        elif isinstance(batch.get(TransitionKey.ACTION), torch.Tensor):
            action_tensor = batch[TransitionKey.ACTION]
            task_embeddings = task_embeddings.to(device=action_tensor.device, dtype=action_tensor.dtype)
        else:
            task_embeddings = task_embeddings.to(dtype=torch.float32)
        
        if batch.get(TransitionKey.COMPLEMENTARY_DATA) is None:
            batch[TransitionKey.COMPLEMENTARY_DATA] = {}
        
        batch[TransitionKey.COMPLEMENTARY_DATA][self.embedding_key] = task_embeddings
        
        return batch
    
    def transform_features(
        self, features: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:

        if "input" in features:
            features["input"][self.embedding_key] = {
                "shape": (self.embedding_dim,),
                "dtype": "float32",
            }
        
        return features
