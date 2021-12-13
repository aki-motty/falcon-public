
#pragma once
#include "LayerConfig.h"
#include "globals.h"
using namespace std;

class FFNConfig : public LayerConfig
{
public:
  size_t batchSize = 0;
  size_t seq_len = 0;
  size_t d_model = 0;
  size_t d_ff = 0;

  FFNConfig(size_t _batch_size, size_t _seq_len, size_t _d_model, size_t _d_ff)
      : batchSize(_batch_size),
        seq_len(_seq_len),
        d_model(_d_model),
        d_ff(_d_ff),
        LayerConfig("FFN"){};
};
