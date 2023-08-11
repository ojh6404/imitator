import os
from abc import ABC, abstractmethod
import numpy as np
from collections import OrderedDict

from typing import Dict, Optional, Tuple, Union, List
import yaml
from easydict import EasyDict as edict
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchvision import models as vision_models
from torchvision import transforms

from imitator.models.base_nets import MLP, RNN, PositionalEncoding, GPT_Backbone
import imitator.utils.tensor_utils as TensorUtils
from imitator.utils.obs_utils import ObservationEncoder, ImageModality, FloatVectorModality, get_normalize_params
from imitator.utils import file_utils as FileUtils


"""
Policy Networks
flow : ObservationEncoder -> ActorCore[MLP, RNN, ...] -> MLPDecoder
"""


class Actor(ABC, nn.Module):
    """
    Base class for actor networks.
    """
    def __init__(self, cfg: Dict) -> None:
        super(Actor, self).__init__()
        self.cfg = cfg
        self.policy_type = cfg.network.policy.model
        self.action_dim = cfg.actions.dim
        if cfg.actions.normalize:
            action_mean, action_std = get_normalize_params(cfg.actions.min, cfg.actions.max)
        else:
            action_mean, action_std = 0.0, 1.0
        self.action_modality = eval(cfg.actions.modality)(name="actions",shape=cfg.actions.dim, mean=action_mean, std=action_std)

        self.nets = nn.ModuleDict()


    @abstractmethod
    def _build_network(self):
        pass


class MLPActor(Actor):
    def __init__(self, cfg: Dict) -> None:
        super(MLPActor, self).__init__(cfg)
        self.mlp_input_dim = sum(
            [cfg.obs[key].obs_encoder.output_dim for key in cfg.obs.keys()]
        ) # sum of all obs_encoder output dims
        self.mlp_kwargs = cfg.network.policy.get("kwargs", {})
        self.mlp_layer_dims = cfg.network.policy.mlp_decoder.layer_dims
        self.mlp_activation = eval("nn." + cfg.network.policy.mlp_decoder.get("activation", "ReLU"))
        self.decoder_output_activation = nn.Tanh if cfg.network.policy.mlp_decoder.get("squash_output", False) else None

        self._build_network()

    def _build_network(self) -> None:
        """
        Build the network.
        inputs passed to obs_encoder -> rnn -> mlp_decoder
        """

        self.nets["obs_encoder"] = ObservationEncoder(self.cfg.obs)
        self.nets["mlp_decoder"] = MLP(
            input_dim=self.mlp_input_dim,
            layer_dims=self.mlp_layer_dims,
            output_dim=self.action_dim,
            activation=self.mlp_activation,
            output_activation=self.decoder_output_activation,
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor],unnormalize: bool = False,) -> torch.Tensor:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = self.nets["obs_encoder"](obs_dict)
        outputs = self.nets["mlp_decoder"](obs_latents)

        if unnormalize:
            outputs = self.action_modality.unprocess_obs(outputs)
        return outputs

    def forward_step(
            self, obs_dict: Dict[str, torch.Tensor], unnormalize: bool = False
    ) -> torch.Tensor:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        outputs = self.forward(obs_dict, unnormalize=unnormalize)
        return outputs


class RNNActor(Actor):
    def __init__(self, cfg: Dict) -> None:
        super(RNNActor, self).__init__(cfg)
        self.rnn_type = cfg.network.policy.rnn.type
        self.rnn_num_layers = cfg.network.policy.rnn.rnn_num_layers
        self.rnn_hidden_dim = cfg.network.policy.rnn.rnn_hidden_dim
        self.rnn_input_dim = sum(
            [cfg.obs[key].obs_encoder.output_dim for key in cfg.obs.keys()]
        ) # sum of all obs_encoder output dims
        self.rnn_kwargs = cfg.network.policy.rnn.get("kwargs", {})

        # for rnn decoder
        self.mlp_layer_dims = cfg.network.policy.mlp_decoder.layer_dims
        self.mlp_activation = eval("nn." + cfg.network.policy.mlp_decoder.get("activation", "ReLU"))
        self.decoder_output_activation = nn.Tanh if cfg.network.policy.mlp_decoder.get("squash_output", False) else None

        self._build_network()


    def _build_network(self) -> None:
        """
        Build the network.
        inputs passed to obs_encoder -> rnn -> mlp_decoder
        """
        self.nets["obs_encoder"] = ObservationEncoder(self.cfg.obs)
        self.nets["mlp_decoder"] = MLP(
            input_dim=self.rnn_hidden_dim,
            layer_dims=self.mlp_layer_dims,
            output_dim=self.action_dim,
            activation=self.mlp_activation,
            output_activation=self.decoder_output_activation,
        )
        self.nets["rnn"] = RNN(
            rnn_input_dim=self.rnn_input_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            rnn_num_layers=self.rnn_num_layers,
            rnn_type=self.rnn_type,
            rnn_kwargs=self.rnn_kwargs,
            per_step_net=self.nets["mlp_decoder"],
        )



    def get_rnn_init_state(self, batch_size: int, device: torch.device) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.nets["rnn"].get_rnn_init_state(batch_size, device)

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        rnn_state: Optional[torch.Tensor] = None,
        return_rnn_state: bool = False,
        unnormalize: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = self.nets["obs_encoder"](obs_dict)
        outputs, rnn_state = self.nets["rnn"](
            inputs=obs_latents, rnn_state=rnn_state, return_rnn_state=True
        )

        if unnormalize:
            outputs = self.action_modality.unprocess_obs(outputs)

        if return_rnn_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward_step(
            self, obs_dict: Dict[str, torch.Tensor], rnn_state: torch.Tensor, unnormalize: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = self.nets["obs_encoder"](obs_dict)
        outputs, rnn_state = self.nets["rnn"].forward_step(obs_latents, rnn_state)
        if unnormalize:
            outputs = self.action_modality.unprocess_obs(outputs)
        return outputs, rnn_state


class TransformerActor(Actor):
    """
    Actor with Transformer encoder and MLP decoder
    """
    def __init__(
        self,
        cfg: Dict,
    ) -> None:
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.
            transformer_embed_dim (int): dimension for embeddings used by transformer
            transformer_num_layers (int): number of transformer blocks to stack
            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is
                computed over this many partitions of the embedding dimension separately.
            transformer_context_length (int): expected length of input sequences
            transformer_activation: non-linearity for input and output layers used in transformer
            transformer_emb_dropout (float): dropout probability for embedding inputs in transformer
            transformer_attn_dropout (float): dropout probability for attention outputs for each transformer block
            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block
            encoder_kwargs (dict): observation encoder config
        """
        super(TransformerActor, self).__init__(cfg)
        self.transformer_type = cfg.network.policy.transformer.type
        self.transformer_num_layers = cfg.network.policy.transformer.transformer_num_layers
        self.transformer_embed_dim = cfg.network.policy.transformer.transformer_embed_dim
        self.transformer_input_dim = sum(
            [cfg.obs[key].obs_encoder.output_dim for key in cfg.obs.keys()]
        ) # sum of all obs_encoder output dims
        self.transformer_kwargs = cfg.network.policy.transformer.get("kwargs", {})
        self.context_length = cfg.network.policy.transformer.context_length

        self.transformer_embed_dropout = 0.1
        self.max_timestep = self.context_length

        # for transformer decoder
        self.mlp_layer_dims = cfg.network.policy.mlp_decoder.layer_dims
        self.mlp_activation = eval("nn." + cfg.network.policy.mlp_decoder.get("activation", "ReLU"))
        self.decoder_output_activation = nn.Tanh if cfg.network.policy.mlp_decoder.get("squash_output", False) else None

        self.freeze = cfg.network.policy.get("freeze", True)

        self._build_network()



    def _build_network(self) -> None:
        """
        Build the network.
        inputs passed to obs_encoder -> transformer -> mlp_decoder
        """
        self.params = nn.ParameterDict()
        self.params["embed_timestep"] = nn.Parameter(
            torch.zeros(1, self.max_timestep, self.transformer_embed_dim)
        ) # TODO : pos enc or something more

        self.nets["obs_encoder"] = ObservationEncoder(self.cfg.obs) #
        self.nets["embedding"] = MLP(
            input_dim=self.transformer_input_dim,
            layer_dims=[],
            output_dim=self.transformer_embed_dim,
        )
        # self.nets["embedding"] = nn.Linear(self.transformer_input_dim, self.transformer_embed_dim)

        self.nets["embed_ln"] = nn.LayerNorm(self.transformer_embed_dim)
        self.nets["embed_drop"] = nn.Dropout(self.transformer_embed_dropout)

        self.nets["transformer"] = GPT_Backbone(
            embed_dim=self.transformer_embed_dim,
            context_length=self.context_length,
            attn_dropout=0.1,
            block_output_dropout=0.1,
            num_layers=6,
            num_heads=8,
            activation="gelu",
        )

        self.nets["mlp_decoder"] = MLP(
            input_dim=self.transformer_embed_dim,
            layer_dims=self.mlp_layer_dims,
            output_dim=self.action_dim,
            activation=self.mlp_activation,
            output_activation=self.decoder_output_activation,
        )

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """

        time_embeddings = self.params["embed_timestep"]
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to transformer,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to transformer backbone.
        """
        # embeddings = self.nets["embed_encoder"](inputs)
        embeddings = self.nets["embedding"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        unnormalize: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        obs_latents = self.nets["obs_encoder"](obs_dict) # [B, T, D] falttened
        assert obs_latents.ndim == 3  # [B, T, D]

        transformer_embeddings = self.input_embedding(obs_latents)
        transformer_encoder_outputs = self.nets["transformer"].forward(transformer_embeddings)
        outputs = self.nets["mlp_decoder"](transformer_encoder_outputs)

        if unnormalize:
            outputs = self.action_modality.unprocess_obs(outputs)

        return outputs

    def forward_step(
            self, obs_dict: Dict[str, torch.Tensor], unnormalize: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        outputs = self.forward(obs_dict, unnormalize=unnormalize) # [B, T, D]
        return outputs[:, -1, :] # [B, D], last timestep cause it uses framestack
