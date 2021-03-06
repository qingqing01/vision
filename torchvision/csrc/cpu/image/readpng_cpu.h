#pragma once

// Comment
#include <torch/torch.h>
#include <string>

C10_EXPORT torch::Tensor decodePNG(
    const torch::Tensor& data,
    int64_t channels = 0);
