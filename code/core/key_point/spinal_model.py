import torch


class SpinalModelBase(torch.nn.Module):
    def forward(self, heatmaps: torch.Tensor):
        """

        :param heatmaps: (num_batch, num_points, height, width)
        :return: (num_batch, num_points, 2)
        """
        size = heatmaps.size()
        flatten = heatmaps.flatten(start_dim=2)
        max_indices = torch.argmax(flatten, dim=-1)
        height_indices = max_indices.flatten() // size[3]
        width_indices = max_indices.flatten() % size[3]
        # 粗略估计
        preds = torch.stack([width_indices, height_indices], dim=1)
        preds = preds.reshape(flatten.shape[0], flatten.shape[1], 2)
        return preds
