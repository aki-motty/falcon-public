
#pragma once
#include "globals.h"
#include <cmath>
using namespace std;

class EncoderLayerConfig
{
public:
  size_t num_heads = 0;
  size_t seq_len = 0;
  size_t d_model = 0;
  size_t batchSize = 0;
  size_t d_ff = 0;

  EncoderLayerConfig(size_t _num_heads, size_t _seq_len, size_t _d_model, size_t _batchSize, size_t _d_ff)
      : num_heads(_num_heads),
        seq_len(_seq_len),
        d_model(_d_model),
        batchSize(_batchSize),
        d_ff(_d_ff)
  {
  };
};