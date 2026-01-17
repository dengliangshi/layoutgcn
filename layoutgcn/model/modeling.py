#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import math
import logging
from typing import Optional, List
from dataclasses import dataclass

# Third-party libraries
import torch
import torchvision
import safetensors.torch
from transformers.modeling_outputs import ModelOutput
from transformers import EfficientNetModel, EfficientNetConfig

# User define module
from .configuration import LayoutGCNConfig

# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------Main-----------------------------------------------------------
class TextEncoder(torch.nn.Module):
    """
    TextEncoder is a PyTorch module implementing a TextCNN-based model for text feature extraction. It processes input text sequences and generates both sequence-level and pooled representations.
    """
    def __init__(self, config):
        """TextCNN model for text feature extraction.
        Args:
            config: Configuration object with the following attributes:
                vocab_size: Size of the vocabulary.
                embedding_dim: Dimension of the word embeddings.
                padding_idx: Index of the padding token.
                max_seq_length: Length of the input sequences.
                num_filters: List of number of filters for each convolutional layer.
                filter_sizes: List of filter sizes for each convolutional layer.
        """
        super(TextEncoder, self).__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = torch.nn.Embedding(
            num_embeddings=config.num_word_embeddings,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_token_idx,
            sparse=False
        )

        # Convolutional layers
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(
                in_channels=config.embedding_dim,
                out_channels=num_filter, 
                kernel_size=filter_size,
                stride=1,
                padding="same"
            ) for num_filter, filter_size in zip(config.num_filters, config.filter_sizes)
        ])
        # Linear projection layer
        self.linear_project = torch.nn.Linear(
            in_features=sum(config.num_filters),
            out_features=config.hidden_size
        )
        # Layer normalization and dropout
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=config.hidden_size)
        # Dropout layer
        self.dropout = torch.nn.Dropout(config.dropout_prob)
        # Cache for sequence mask generation
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_seq_length, dtype=torch.long)
        )

    def conv_and_pool(self, embeddings, sequence_mask):
        """Convolution and max-pooling layer.
            Args:
                embeddings: Embedded input sequences, shape [batch_size, max_seq_length, embedding_dim]
                sequence_mask: Mask for sequence, shape [batch_size, max_seq_length]
            Returns:
                sequence_outputs: Sequence-level outputs, shape [batch_size, max_seq_length, hidden_size]
                pooling_outputs: Pooled outputs, shape [batch_size, hidden_size]
        """
        # [batch_size, embedding_dim, max_seq_length]
        trans_embeddings = embeddings.transpose(1, 2)  
        ngram_list = [torch.nn.functional.relu(conv(trans_embeddings), inplace=True)
            for conv in self.convs]
        # [batch_size, sum(num_filters), max_seq_length]
        ngrams = torch.cat(ngram_list, dim=1)
        # [batch_size, max_seq_length, sum(num_filters)]
        trans_ngrams = ngrams.transpose(1, 2) * sequence_mask.unsqueeze(-1)
        # [batch_size, max_seq_length, hidden_size]
        sequence_outputs = self.linear_project(trans_ngrams)
        # Max pooling over time
        # [batch_size, sum(num_filters)]
        masked_ngrams = ngrams + (1 - sequence_mask.unsqueeze(1)) * -1e9
        pooling_features = masked_ngrams.max(dim=-1, keepdim=False)[0]
        # [batch_size, hidden_size]
        pooling_outputs = self.linear_project(pooling_features)
        return sequence_outputs, pooling_outputs

    def forward(self, token_ids, sequence_lengths):
        """Forward pass of the model.
        Args:
            token_ids: Input token ids, shape [batch_size, max_seq_length].
            sequence_lengths: Length of each sequence, shape [batch_size].
        Returns:
            sequence_outputs: Sequence-level outputs, shape [batch_size, max_seq_length, hidden_size]
            pooling_outputs: Pooled outputs, shape [batch_size, hidden_size]
            sequence_mask: Mask for sequence, shape [batch_size, max_seq_length]
        """
        # Convert sparse tensor to dense if necessary
        if token_ids.is_sparse:
            token_ids = token_ids.to_dense()
        sequence_mask = (self.position_ids.unsqueeze(0) < sequence_lengths.unsqueeze(1)).to(torch.int64)
        # [batch_size, max_seq_length, embedding_dim]
        embeddings = self.embedding(token_ids) * sequence_mask.unsqueeze(-1)
        # sequence_outputs: [batch_size, max_seq_length, hidden_size]
        # pooling_outputs: [batch_size, hidden_size]
        sequence_outputs, pooling_outputs = self.conv_and_pool(embeddings, sequence_mask)
        # Layer normalization
        normalized_pooling_outputs = self.layer_norm(pooling_outputs)
        # Dropout
        dropped_pooling_outputs = self.dropout(normalized_pooling_outputs)
        return sequence_outputs, dropped_pooling_outputs, sequence_mask


class VisualEncoder(torch.nn.Module):
    """EfficentNet model for image feature extraction."""
    
    def __init__(self, config):
        super(VisualEncoder, self).__init__()
        self.config = config
        # EfficientNet model
        self.efficientnet_config = EfficientNetConfig.from_pretrained(
            pretrained_model_name_or_path=config.efficientnet_model_path
        )
        self.efficientnet = EfficientNetModel.from_pretrained(
            pretrained_model_name_or_path=config.efficientnet_model_path,
            config=self.efficientnet_config
        )
        self.lp_for_image_rep = torch.nn.Linear(
            in_features=self.efficientnet_config.out_channels,
            out_features=config.hidden_size
        )
        # Linear projection layer
        self.linear_project = torch.nn.Linear(
            in_features=sum(self.efficientnet_config.out_channels),
            out_features=2 * config.hidden_size
        )
        # Layer normalization and dropout
        self.layer_norm = torch.nn.LayerNorm(2 * self.config.hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

        # Pre-compute block indices for efficiency
        self.block_indices = self._compute_block_indices()

        # Cache for spatial scales
        self.register_buffer(    
            "spatial_scale_base",
            torch.tensor(1.0 / self.efficientnet_config.image_size)
        )

    def _compute_block_indices(self) -> List[int]:
        """Compute block indices for efficientnet."""
        indices = []
        cum_sum = 0
        for num_blocks in self.efficientnet_config.num_block_repeats:
            cum_sum += num_blocks
            indices.append(cum_sum)
        return indices

    def forward(self, piexel_values, rois):
        """Forward pass of the model."""
        index = 0
        roi_pooling_outputs = []

        # Extract features from EfficientNet
        features = self.efficientnet(
            pixel_values=piexel_values,
            output_hidden_states=True,
            return_dict=True
        )
        # RoI pooling for each block's output
        for num_block in self.efficientnet_config.num_block_repeats:
            hidden_state = features.hidden_states[index+num_block]
            spatial_scale = hidden_state.shape[-1] / self.efficientnet_config.image_size
            roi_pooling_output = torchvision.ops.roi_pool(
                input=hidden_state,
                boxes=rois,
                output_size=(self.config.roi_pooling_size, self.config.roi_pooling_size),
                spatial_scale=spatial_scale
            )
            roi_pooling_outputs.append(roi_pooling_output)
            index += num_block
        # Pooling output from the last block, [batch_size, hidden_dim]
        pooling_output = self.lp_for_image_rep(features.pooler_output)
        concat_roi_pooling_outputs = torch.cat(roi_pooling_outputs, dim=1)
        mean_roi_pooling_outputs = torch.mean(concat_roi_pooling_outputs, dim=[2, 3])
        # Node features, [batch_size, max_num_nodes, 2*hidden_size]
        roi_pooling_features = self.linear_project(mean_roi_pooling_outputs)
        reshaped_roi_pooling_features = roi_pooling_features.reshape((-1, self.config.max_num_nodes, 2*self.config.hidden_size))
        normalized_features = self.layer_norm(reshaped_roi_pooling_features)
        dropped_features = self.dropout(normalized_features)
        return pooling_output, dropped_features


class LayoutEncoder(torch.nn.Module):

    """Layout feature encoder."""
    
    def __init__(self, config):
        super(LayoutEncoder, self).__init__()
        self.config = config
        # Linear projection layer
        self.linear_project = torch.nn.Linear(
            in_features=7,
            out_features=config.hidden_size
        )
        # Layer normalization and dropout
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, layout_features):
        """Forward pass of the model.
        args:
            layout_features: [batch_size, max_num_nodes, 7]
        return:
            [batch_size, max_num_nodes, hidden_size]
        """
        outputs = self.linear_project(layout_features)
        normalized_outputs = self.layer_norm(outputs)
        dropped_outputs = self.dropout(normalized_outputs)
        return dropped_outputs


class GraphConvolution(torch.nn.Module):
    """Graph Convolutional Layer."""
    
    def __init__(self, in_features, out_features, bias=True):
        """"""
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output += self.bias
        return torch.nn.functional.relu(output, inplace=True)


class GCN(torch.nn.Module):
    """Graph Convolutional Network for node classification."""
    
    def __init__(self, config):
        super(GCN, self).__init__()
        self.config = config
        # Angle embedding layer
        self.angle_embeddings = torch.nn.Embedding(
            num_embeddings=config.num_angle_embeddings,
            embedding_dim=config.embedding_dim
        )
        # Layers for computing adjacency matrix from angle embeddings
        self.first_linear = torch.nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.hidden_size
        )
        # Second linear layer to produce a single value
        self.second_linear = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=1
        )
        # Learnable parameter for weighting the relation adjacency matrix
        self.eta = torch.nn.Parameter(torch.ones(1))
        # Graph convolutional layers
        self.gc1 = GraphConvolution(2*config.hidden_size, config.hidden_size)
        #  Second graph convolutional layer
        self.gc2 = GraphConvolution(config.hidden_size, config.hidden_size)
        # Attention pooling layer
        self.attention_linear = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=1
        )
        # Layer normalization and dropout
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        rowsum = torch.sum(adj, dim=-1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    def adj_angle_matrix(self, adj_a):
        angle_embeddings = self.angle_embeddings(adj_a)
        hidden_states = torch.nn.functional.relu(self.first_linear(angle_embeddings), inplace=True)
        outputs =  torch.sigmoid(self.second_linear(hidden_states))
        return outputs.squeeze()

    def attention_pooling(self, x):
        """Attention pooling layer to get graph-level representation."""
        attn_weights = torch.nn.functional.softmax(self.attention_linear(x), dim=1)
        graph_embedding = torch.sum(attn_weights * x, dim=1)
        return graph_embedding

    def forward(self, x, adj_radical_dist, adj_angle, graph_mask):
        """Forward pass of the GCN.
        Args:
            x: Node features, shape (batch_size, max_num_nodes, embedding_dim)
            adj: Adjacency matrix, shape (batch_size, max_num_nodes, max_num_nodes)
            Returns
            x: Output features, shape (batch_size, max_num_nodes, num_features)
        """
        # Compute adjacency matrix from angle embeddings, [batch_size, max_num_nodes, max_num_nodes]
        adj_a = self.adj_angle_matrix(adj_angle)
        # Normalize adjacency matrix, [batch_size, max_num_nodes, max_num_nodes]
        adj = self.normalize_adj((adj_a + self.eta * adj_radical_dist) * graph_mask + 1e-5)
        # [batch_size, max_num_nodes, hidden_size]
        node_embeddings = self.gc2(self.gc1(x, adj), adj)
        # Graph-level representation via attention pooling, [batch_size, hidden_size]
        graph_embeddings = self.attention_pooling(node_embeddings)
        return node_embeddings, graph_embeddings


class LayoutGCNModel(torch.nn.Module):
    def __init__(self, config: LayoutGCNConfig):
        super(LayoutGCNModel, self).__init__()
        self.config = config
        # Layout encoder
        self.layout_encoder = LayoutEncoder(config)
        # Text encoder
        self.text_cnn = TextEncoder(config)
        # Visual encoder
        if self.config.use_image:
            self.gate = torch.nn.Linear(
                in_features=4*config.hidden_size,
                out_features=1
            )
            self.visual_encoder = VisualEncoder(config)
        # Graph convolutional network
        self.gcn = GCN(config)
        # Cache for sequence mask generation
        self.register_buffer(
            "position_ids",
            torch.arange(self.config.max_num_nodes, dtype=torch.long)
        )

    def forward(self,
            token_ids: Optional[torch.Tensor] = None,
            sequence_lengths: Optional[torch.Tensor] = None,
            layout_features: Optional[torch.Tensor] = None,
            num_nodes: Optional[torch.Tensor] = None,
            adj_radical_dist: Optional[torch.Tensor] = None,
            adj_angle: Optional[torch.Tensor] = None,
            pixel_values: Optional[torch.Tensor] = None,
            rois: Optional[torch.Tensor] = None
        ):
        """Forward pass of the model.
        args:
            token_ids: Input token ids, shape [batch_size, max_num_nodes, max_seq_length].
            sequence_lengths: Length of each sequence, shape [batch_size, max_num_nodes].
            layout_features: Layout features, shape (batch_size, max_num_nodes, num_features)
            adj_r: Adjacency matrix for relation graph, shape (batch_size, max_num_nodes, max_num_nodes)
            adj_a: Adjacency matrix for attention graph, shape (batch_size, max_num_nodes, max_num_nodes)
            pixel_values: Input pixel values, shape (batch_size, 3, height, width)
            rois: Regions of interest for RoI pooling, shape (batch_size, max_num_nodes, 4).
        returns:
            gcn_output: Output features from GCN, shape (batch_size, max_num_nodes, num_features)
        """
        device = token_ids.device
        # (batch_size, max_num_nodes, sum(num_filters))
        flatten_token_ids = token_ids.view(-1, self.config.max_seq_length)
        graph_range = self.position_ids.unsqueeze(0).expand(num_nodes.shape[0], self.config.max_num_nodes)
        graph_mask = (graph_range < num_nodes.unsqueeze(1)).to(torch.int64)
        flatten_sequence_lengths = sequence_lengths.view(-1)[graph_mask.view(-1) > 0]
        # Text encoder
        sequence_outputs, pooling_outputs, sequence_mask = self.text_cnn(
            token_ids=flatten_token_ids[graph_mask.reshape(-1) > 0],
            sequence_lengths=flatten_sequence_lengths
        )
        flatten_node_textual_features = torch.zeros(flatten_token_ids.shape[0], self.config.hidden_size).to(device)
        flatten_node_textual_features[graph_mask.reshape(-1) > 0] = pooling_outputs
        node_textual_features = flatten_node_textual_features.view(token_ids.shape[0], token_ids.shape[1], -1)
        # Layout encoder
        node_layout_features = self.layout_encoder(layout_features)
        node_features = torch.cat((node_textual_features, node_layout_features), dim=-1)
        # (batch_size, max_num_nodes, roi_pooling_size*roi_pooling_size*efficientnet_hidden_size)
        if self.config.use_image:
            image_features, node_visual_features = self.visual_encoder(pixel_values, rois)
            gate = torch.sigmoid(self.gate(torch.cat((node_features, node_visual_features), dim=-1)))
            node_features = node_features + gate * node_visual_features
        masked_node_features = node_features * graph_mask.unsqueeze(-1)
        graph_adj_mask = graph_mask.unsqueeze(2) & graph_mask.unsqueeze(1)
        node_embeddings, graph_embedding = self.gcn(masked_node_features, adj_radical_dist, adj_angle, graph_adj_mask)
        node_embeddings = torch.cat((node_embeddings, masked_node_features), dim=-1)
        # Document embedding, [batch_size, 2*hidden_size]
        if self.config.use_image:
            document_embedding = torch.cat([graph_embedding, image_features], dim=-1)
        else:
            document_embedding = graph_embedding
        return {
            "sequence_outputs": sequence_outputs,
            "sequence_mask": sequence_mask,
            "node_embeddings": node_embeddings,
            "graph_mask": graph_mask,
            "graph_adj_mask": graph_adj_mask,
            "document_embedding": document_embedding
        }


class LayoutGCNBaseModel(torch.nn.Module):

    def __init__(self, config: LayoutGCNConfig):
        super(LayoutGCNBaseModel, self).__init__()
        self.config = config
        self.model = LayoutGCNModel(config)

    @classmethod
    def load_from_model_path(cls, model_path: str):
        config = LayoutGCNConfig.from_model_path(model_path)
        model = cls(config)
        model_dict = safetensors.torch.load_file(os.path.join(model_path, "model.safetensors"))
        model.load_state_dict(model_dict)
        return model


@dataclass
class DocumentClassifierOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args: (when not in `return_dict=True` mode):
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when ` labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`torch.FloatTensor` of shape `(batch_size, 2*hidden_size)`):
            Document embedding.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probabilities: torch.FloatTensor = None


class LayoutGCNForDocClassification(LayoutGCNBaseModel):
    
    def __init__(self, config):
        super(LayoutGCNForDocClassification, self).__init__(config)
        self.classifier = torch.nn.Linear(
            in_features=2 * config.hidden_size if config.use_image else config.hidden_size,
            out_features=config.num_labels
        )
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self,
        token_ids: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        layout_features: Optional[torch.Tensor] = None,
        num_nodes: Optional[torch.Tensor] = None,
        adj_radical_dist: Optional[torch.Tensor] = None,
        adj_angle: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        rois: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None
    ):
        # get the document embedding
        outputs = self.model(token_ids, sequence_lengths, layout_features,
            num_nodes, adj_radical_dist, adj_angle, pixel_values, rois)
        dropped_document_embedding = self.dropout(outputs["document_embedding"])
        logits = self.classifier(dropped_document_embedding)
        probabilities = self.softmax(logits)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        else:
            loss = None

        if not return_dict:
            output = (logits)
            return ((loss,) + output) if loss is not None else output

        return DocumentClassifierOutput(
            loss=loss,
            logits=logits,
            probabilities=probabilities
        )


@dataclass
class NodeClassifierOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args: (when not in `return_dict=True` mode):
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when ` labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`torch.FloatTensor` of shape `(batch_size, 2*hidden_size)`):
            Document embedding.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probabilities: Optional[torch.FloatTensor] = None
    mask: Optional[torch.FloatTensor] = None


class LayoutGCNForNodeClassification(LayoutGCNBaseModel):
    
    def __init__(self, config):
        super(LayoutGCNForNodeClassification, self).__init__(config)
        self.classifier = torch.nn.Linear(3*config.hidden_size, config.num_labels)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, 
        token_ids: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        layout_features: Optional[torch.Tensor] = None,
        num_nodes: Optional[torch.Tensor] = None,
        adj_radical_dist: Optional[torch.Tensor] = None,
        adj_angle: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        rois: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None
    ):
        loss = None
        batch_size = token_ids.shape[0]
        output = self.model(token_ids, sequence_lengths, layout_features,
            num_nodes, adj_radical_dist, adj_angle, pixel_values, rois)
        # [batch_size, max_num_nodes, num_labels]
        logits = self.classifier(output["node_embeddings"])
        reshaped_logits = logits.reshape(batch_size, self.config.max_num_nodes, -1)
        probabilities = torch.softmax(reshaped_logits, dim=-1)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        else:
            loss = None

        if not return_dict:
            output = (reshaped_logits, probabilities, output["graph_mask"],)
            return ((loss,) + output) if loss is not None else output

        return NodeClassifierOutput(
            loss=loss,
            logits=reshaped_logits,
            probabilities=probabilities,
            mask=output["graph_mask"]
        )


@dataclass
class InfoExtractionOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args: (when not in `return_dict=True` mode):
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when ` labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`torch.FloatTensor` of shape `(batch_size, 2*hidden_size)`):
            Document embedding.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probabilities: Optional[torch.FloatTensor] = None
    mask: torch.FloatTensor = None


class LayoutGCNForInfoExtraction(LayoutGCNBaseModel):
    
    def __init__(self, config):
        super(LayoutGCNForInfoExtraction, self).__init__(config)
        self.linear = torch.nn.Linear(4*config.hidden_size, config.num_labels)
    def forward(self, 
        token_ids: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        layout_features: Optional[torch.Tensor] = None,
        num_nodes: Optional[torch.Tensor] = None,
        adj_radical_dist: Optional[torch.Tensor] = None,
        adj_angle: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        rois: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None
    ):
        output = self.model(token_ids, sequence_lengths, layout_features,
            num_nodes, adj_radical_dist, adj_angle, pixel_values, rois)
        # [batch_size * max_num_nodes, 3*hidden_size]
        flatten_node_embeddings = output["node_embeddings"].view((-1, output["node_embeddings"].shape[-1]))
        node_embeddings = flatten_node_embeddings[output["graph_mask"].view(-1) > 0]
        tiled_node_embeddings = torch.tile(node_embeddings.unsqueeze(1) , dims=(1, self.config.max_seq_length, 1))

        # [total_num_nodes, max_seq_length, 4*hidden_size]
        enhanced_sequence_outpus = torch.concatenate([output["sequence_outputs"], tiled_node_embeddings], dim=-1)
        logits = self.linear(enhanced_sequence_outpus)

        probabilities = torch.softmax(logits, dim=-1)

        padding_probabilities = torch.zeros(
            size=(flatten_node_embeddings.shape[0], self.config.max_seq_length, self.config.num_labels),
            dtype=torch.float32,
            device=logits.device
        )
        padding_probabilities[output["graph_mask"].view(-1) > 0] = probabilities
        reshaped_padding_probabilities = padding_probabilities.view(-1, self.config.max_num_nodes,
            self.config.max_seq_length, self.config.num_labels)

        padding_sequence_mask = torch.zeros(
            size=(flatten_node_embeddings.shape[0], self.config.max_seq_length),
            dtype=torch.int64,
            device=logits.device
        )
        padding_sequence_mask[output["graph_mask"].view(-1) > 0] = output["sequence_mask"]
        reshaped_padding_sequence_mask = padding_sequence_mask.view(-1, self.config.max_num_nodes, self.config.max_seq_length)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            labels_wo_padding = labels.reshape((-1, self.config.max_seq_length))[output["graph_mask"].view(-1) > 0]
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels_wo_padding.view(-1))
        else:
            loss = None

        if not return_dict:
            output = (logits, reshaped_padding_probabilities, reshaped_padding_sequence_mask, )
            return ((loss,) + output) if loss is not None else output

        return InfoExtractionOutput(
            loss=loss,
            logits=logits,
            probabilities=reshaped_padding_probabilities,
            mask=reshaped_padding_sequence_mask
        )


@dataclass
class LinkPredictionOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.
    Args: (when not in `return_dict=True` mode):
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when ` labels` is provided):
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`torch.FloatTensor` of shape `(batch_size, 2*hidden_size)`):
            Document embedding.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    probabilities: torch.FloatTensor = None
    mask: Optional[torch.FloatTensor] = None


class LayoutGCNForLinkPrediction(LayoutGCNBaseModel):

    def __init__(self, config):
        super(LayoutGCNForLinkPrediction, self).__init__(config)
        if config.directed_link:
            self.transform = torch.nn.Parameter(3*config.hidden_size, 3*config.hidden_size)
        else:
            self.transform = torch.nn.Parameter(2*config.hidden_size, 1)
        self.dropout = torch.nn.Dropout(config.dropout_prob)

    def forword(self, 
        token_ids: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        layout_features: Optional[torch.Tensor] = None,
        num_nodes: Optional[torch.Tensor] = None,
        adj_radical_dist: Optional[torch.Tensor] = None,
        adj_angle: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        rois: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None
    ):
        """Forward pass for link prediction.
    
        Args:
            token_ids: Input token ids, shape [batch_size, max_num_nodes, max_seq_length].
            sequence_lengths: Length of each sequence, shape [batch_size, max_num_nodes].
            layout_features: Layout features, shape (batch_size, max_num_nodes, 7)
            num_nodes: Number of nodes per graph, shape (batch_size,)
            adj_radical_dist: Adjacency matrix for radical distance graph, shape (batch_size, max_num_nodes, max_num_nodes)
            adj_angle: Adjacency matrix for angle graph, shape (batch_size, max_num_nodes, max_num_nodes)
            pixel_values: Input pixel values, shape (batch_size, 3, height, width)
            rois: Regions of interest for RoI pooling, shape (batch_size, max_num_nodes, 4).
            labels: Ground truth labels for link prediction, shape (batch_size, max_num_nodes, max_num_nodes)
            return_dict: Whether to return a dictionary or tuple
        
        Returns:
            LinkPredictionOutput or tuple containing loss, logits, and mask
        """
        # Get node embeddings from the model
        output = self.model(token_ids, sequence_lengths, layout_features,
            num_nodes, adj_radical_dist, adj_angle, pixel_values, rois)
        
        if self.config.directed_link:
            transform_node_embeddings = torch.matmul(output["node_embeddings"], self.transform)
            logits = torch.matmul(transform_node_embeddings, transform_node_embeddings.transpose(1, 2))
        else:
            left = output["node_embeddings"].unsqueeze(2).expand(-1, -1, num_nodes, -1)
            right = output["node_embeddings"].unsqueeze(1).expand(-1, num_nodes, -1, -1)
            logits = torch.matmul(left * right, self.transform)
        probabilities = torch.softmax(logits, dim=-1)

        if labels is not None:
            sampling_mask = self._n_sampling_mask(logits, labels, output["graph_adj_mask"])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                input=logits.view(-1, 1),
                target=labels.view(-1, 1),
                weight=sampling_mask.view(-1, 1)
            )
        else:
            loss = None

        if not return_dict:
            output = (logits, probabilities, output["graph_adj_mask"])
            return ((loss,) + output) if loss is not None else output
        
        return LinkPredictionOutput(
            loss=loss,
            logits=logits,
            probabilities=probabilities,
            mask=output["graph_adj_mask"]
        )
