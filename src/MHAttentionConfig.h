
#pragma once
#include "globals.h"
#include <cmath>
using namespace std;

class MHAttentionConfig
{
public:
	size_t num_heads = 0;
    size_t seq_len = 0;
    size_t d_model = 0;
    size_t depth = 0;
    size_t nb_features = 0;
    size_t batchSize = 0;
    bool causal = 0;
    double attention_normalizer = 0.0001;
    size_t seed = 0;

	MHAttentionConfig(size_t _num_heads, size_t _seq_len, size_t _d_model, size_t _batchSize, bool _causal, double _attention_normalizer, size_t _seed)
	:num_heads(_num_heads),
	 seq_len(_seq_len),
	 d_model(_d_model),
     batchSize(_batchSize),
	 causal(_causal),
     attention_normalizer(_attention_normalizer),
     seed(_seed)
	{
		assert(d_model % this->num_heads == 0 && "d_model % num_heads != 0");
        this->depth = this->d_model / this->num_heads;
        this->nb_features = static_cast<size_t>(this->depth * log(this->depth));
	};
};
