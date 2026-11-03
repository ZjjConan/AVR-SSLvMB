# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import torch
import numpy as np


class BongardSampler():

    def __init__(self, n_tasks, n_batch, ep_per_batch=1, seed=123, is_unsupervised=False):
        self.random_state = np.random.RandomState(seed)
        self.n_tasks = n_tasks
        self.n_batch = n_batch
        self.ep_per_batch = ep_per_batch
        self.bong_size = 7
        self.is_unsupervised = is_unsupervised

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            episodes = self.random_state.choice(self.n_tasks, self.ep_per_batch, replace=False)
            if self.is_unsupervised:
                # The last image is always used as the query.
                # This prevents the last image from being used as context during training, avoiding data leakage.
                bong_size = self.bong_size-1
            else:
                bong_size = self.bong_size
            batch = []
            for ep in episodes:
                indices_pos = self.random_state.permutation(range(bong_size))  # permute for varying test images
                if self.is_unsupervised:
                    # do not permute pseudo-test image for unsupervised learning
                    indices_pos = np.append(indices_pos, 6)
                batch.extend([ep * self.bong_size * 2 + i for i in indices_pos])
                indices_neg = self.random_state.permutation(range(bong_size))  # permute for varying test images
                if self.is_unsupervised:
                    # do not permute pseudo-test image for unsupervised learning
                    indices_neg = np.append(indices_neg, 6)
                batch.extend([ep * self.bong_size * 2 + i + self.bong_size for i in indices_neg])

            batch = torch.tensor(batch)  # ep_per_batch * 2 * 7
            yield batch.view(-1)

