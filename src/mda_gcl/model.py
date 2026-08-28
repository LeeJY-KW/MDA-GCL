"""MDA-GCL encoder, projection, classification, and adversarial heads."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GradientReversal(torch.autograd.Function):
    """Identity in the forward pass and scaled gradient reversal backward."""

    @staticmethod
    def forward(ctx: object, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)  # type: ignore[attr-defined]
        return inputs.view_as(inputs)

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None  # type: ignore[attr-defined]


class GraphConvolution(nn.Module):
    """Dense graph convolution ``A X W + b`` for a fold-local graph."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("graph convolution dimensions must be positive")
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must be a 2-D tensor")
        if features.shape[1] != self.weight.shape[0]:
            raise ValueError(
                f"expected {self.weight.shape[0]} input features; received {features.shape[1]}"
            )
        if adjacency.ndim != 2 or adjacency.shape != (
            features.shape[0],
            features.shape[0],
        ):
            raise ValueError("adjacency must be square and match the feature node count")
        output = adjacency @ (features @ self.weight)
        return output if self.bias is None else output + self.bias


class Encoder(nn.Module):
    """The manuscript's two-layer CELU graph encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, output_dim)
        self.activation = nn.CELU()

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.gcn1(features, adjacency))
        return self.activation(self.gcn2(hidden, adjacency))


class MDAGCL(nn.Module):
    """Shared GCN with projection, emotion, and binary domain heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        emotion_classes: int,
        *,
        temperature: float = 0.7,
    ) -> None:
        super().__init__()
        dimensions = (input_dim, hidden_dim, embedding_dim, emotion_classes)
        if any(isinstance(value, bool) or value <= 0 for value in dimensions):
            raise ValueError("model dimensions and emotion_classes must be positive")
        if emotion_classes < 2:
            raise ValueError("emotion_classes must be at least two")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite positive number")

        self.encoder = Encoder(input_dim, hidden_dim, embedding_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.CELU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )
        paired_dim = embedding_dim * 2
        self.emotion_classifier = nn.Linear(paired_dim, emotion_classes)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(paired_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 2),
        )
        self.temperature = float(temperature)
        self.domain_classes = 2
        self.domain_dropout = 0.5

    def encode(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.encoder(features, adjacency)

    def project(self, representations: torch.Tensor) -> torch.Tensor:
        return self.projection_head(representations)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Return one embedding per node from the supplied graph view."""

        return self.project(self.encode(features, adjacency))

    def classify(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Classify concatenated projected view embeddings."""

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2-D tensor")
        if embeddings.shape[1] != self.emotion_classifier.in_features:
            raise ValueError("classification requires two concatenated projected views")
        return self.emotion_classifier(F.celu(embeddings))

    def domain_logits(self, embeddings: torch.Tensor, grl_scale: float) -> torch.Tensor:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.emotion_classifier.in_features:
            raise ValueError("domain classification requires two concatenated projected views")
        if not math.isfinite(grl_scale) or grl_scale < 0:
            raise ValueError("grl_scale must be a finite non-negative number")
        reversed_embeddings = GradientReversal.apply(embeddings, float(grl_scale))
        return self.domain_discriminator(reversed_embeddings)

    def contrastive_loss(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Bidirectional node-level InfoNCE from the manuscript and legacy code."""

        if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
            raise ValueError("contrastive embeddings must be matching 2-D tensors")
        if first.shape[0] == 0:
            raise ValueError("contrastive embeddings must contain at least one node")
        first_normalized = F.normalize(first, dim=1)
        second_normalized = F.normalize(second, dim=1)
        return 0.5 * (
            self._semi_loss(first_normalized, second_normalized)
            + self._semi_loss(second_normalized, first_normalized)
        )

    def _semi_loss(self, anchors: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        same_logits = anchors @ anchors.T / self.temperature
        cross_logits = anchors @ other.T / self.temperature
        diagonal = torch.eye(
            anchors.shape[0], dtype=torch.bool, device=anchors.device
        )
        same_logits = same_logits.masked_fill(diagonal, -torch.inf)
        denominator = torch.logsumexp(
            torch.cat((same_logits, cross_logits), dim=1), dim=1
        )
        positives = cross_logits.diagonal()
        return (denominator - positives).mean()
