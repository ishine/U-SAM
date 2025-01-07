import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreAwarePredictionNetwork(nn.Module):
    def __init__(self, patch_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, patch_features):
        # Predict significance score for each patch
        significance_scores = self.mlp(patch_features)  # Shape: (batch, num_patches, 1)
        return significance_scores.squeeze(-1)  # Shape: (batch, num_patches)


class LanguageContextPatchSelection(nn.Module):
    def __init__(self, patch_dim, hidden_dim):
        super().__init__()
        self.score_network = ScoreAwarePredictionNetwork(patch_dim, hidden_dim)

    def forward(self, patch_features, text_features):
        batch, num_patches, patch_dim = patch_features.size()
        
        # Step 1: Predictive score
        predictive_scores = self.score_network(patch_features)  # Shape: (batch, num_patches)

        # Step 2: Text-relevant and Image-salient scores
        visual_global = patch_features.mean(dim=1, keepdim=True)  # Shape: (batch, 1, patch_dim)
        text_global = text_features.mean(dim=1, keepdim=True)  # Shape: (batch, 1, text_feature_dim)

        text_relevant_scores = F.cosine_similarity(patch_features, text_global, dim=-1)  # Shape: (batch, num_patches)
        image_salient_scores = F.cosine_similarity(patch_features, visual_global, dim=-1)  # Shape: (batch, num_patches)

        # Normalize scores to [0, 1] range
        text_relevant_scores = (text_relevant_scores - text_relevant_scores.min()) / (text_relevant_scores.max() - text_relevant_scores.min())
        image_salient_scores = (image_salient_scores - image_salient_scores.min()) / (image_salient_scores.max() - image_salient_scores.min())

        # Step 3: Final significance score
        beta = 0.5  # Weight parameter
        significance_scores = (1 - beta) * predictive_scores + beta * 0.5 * (text_relevant_scores + image_salient_scores)
        return significance_scores


class SemanticSpatialPatchCalibration(nn.Module):
    def __init__(self, hidden_dim, num_aggregated_patches):
        super().__init__()
        self.aggregation_network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_aggregated_patches),
            nn.Softmax(dim=-1)
        )

    def forward(self, significant_patches, decision_matrix):
        # Select patches based on decision matrix
        selected_patches = significant_patches * decision_matrix.unsqueeze(-1)
        weights = self.aggregation_network(selected_patches)  # Shape: (batch, num_patches, num_aggregated_patches)

        # Aggregate patches based on learned weights
        calibrated_patches = torch.bmm(weights.transpose(1, 2), selected_patches)  # Shape: (batch, num_aggregated_patches, patch_dim)
        return calibrated_patches


class SparsePatchWordAlignment(nn.Module):
    def forward(self, calibrated_patches, text_features):
        # Compute similarity matrix
        similarity_matrix = torch.bmm(calibrated_patches, text_features.transpose(1, 2))  # Shape: (batch, num_aggregated_patches, text_feature_len)
        
        # Patch-to-word and word-to-patch alignment
        patch_to_word_alignment = similarity_matrix.max(dim=-1)[0].mean(dim=-1)
        word_to_patch_alignment = similarity_matrix.max(dim=1)[0].mean(dim=-1)
        
        # Overall alignment score
        alignment_score = 0.5 * (patch_to_word_alignment + word_to_patch_alignment)
        return alignment_score



class LAPS_Loss(nn.Module):
    def __init__(self, margin=0.2, desired_ratio=0.5):
        """
        Initialize the loss function for LAPS framework.
        
        Args:
            margin (float): Margin for the triplet loss.
            desired_ratio (float): Target ratio of selected patches.
        """
        super().__init__()
        self.margin = margin
        self.desired_ratio = desired_ratio
        self.mse_loss = nn.MSELoss()

    def forward(self, alignment_score, positive_score, negative_score, significance_scores):
        """
        Compute the combined loss for LAPS framework.

        Args:
            alignment_score (Tensor): Overall alignment score for positive pairs. Shape: (batch,).
            positive_score (Tensor): Alignment score for positive samples (patch-text pairs).
            negative_score (Tensor): Alignment score for negative samples (patch-text pairs).
            significance_scores (Tensor): Predicted significance scores for patches. Shape: (batch, num_patches).

        Returns:
            Tensor: Combined loss for LAPS framework.
        """
        # Alignment Loss (Bi-directional Triplet Loss with hard negative mining)
        triplet_loss = F.relu(self.margin + negative_score - positive_score).mean()

        # Ratio Constraint Loss (MSE to control the patch selection ratio)
        actual_ratio = significance_scores.mean(dim=-1)  # Shape: (batch,)
        ratio_loss = self.mse_loss(actual_ratio, torch.full_like(actual_ratio, self.desired_ratio))

        # Total Loss
        total_loss = triplet_loss + ratio_loss
        return total_loss


if __name__ == "__main__":
    # Instantiate modules
    patch_dim = 512
    hidden_dim = 512
    num_aggregated_patches = 5  # Adjust based on your configuration

    lps_module = LanguageContextPatchSelection(patch_dim, hidden_dim)
    spc_module = SemanticSpatialPatchCalibration(hidden_dim, num_aggregated_patches)
    spa_module = SparsePatchWordAlignment()

    # Sample input
    batch = 1
    num_patches = 10
    text_feature_len = 10

    patch_features = torch.randn(batch, num_patches, patch_dim)  
    text_features = torch.randn(batch, text_feature_len, patch_dim)  # Adjust text features to match `patch_dim`


    # Instantiate the loss function
    loss_fn = LAPS_Loss(margin=0.2, desired_ratio=0.5)

    # Forward pass through the model components
    significance_scores = lps_module(patch_features, text_features)  # LPS module output
    decision_matrix = (significance_scores > 0.5).float()  # Binary decision matrix based on threshold
    calibrated_patches = spc_module(patch_features, decision_matrix)  # SPC module output
    alignment_score = spa_module(calibrated_patches, text_features)  # SPA module output

    # Generate positive and negative alignment scores for the triplet loss
    # The positive score is the alignment score itself (calculated between calibrated patches and correct text features)
    positive_score = alignment_score

    # To get a negative alignment score, we can simulate it by shuffling `text_features` (to represent irrelevant text features)
    # Note: This shuffling is just one method; depending on your application, you might implement other ways of obtaining negatives
    negative_text_features = text_features[torch.randperm(text_features.size(0))]  # Shuffled batch for negative example
    negative_score = spa_module(calibrated_patches, negative_text_features)  # SPA module output with negative pairs

    # Calculate the total LAPS loss
    total_loss = loss_fn(alignment_score, positive_score, negative_score, significance_scores)

    # Print the loss value (optional, for debugging)


    print("Significance Scores:", significance_scores.shape, significance_scores)
    print("Decision Matrix:", decision_matrix.shape, decision_matrix)
    print("Calibrated Patches:", calibrated_patches.shape)
    print("Alignment Score:", alignment_score.shape, alignment_score)

    print("Total LAPS Loss:", total_loss)
