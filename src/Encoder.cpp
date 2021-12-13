#pragma once
#include "Encoder.h"
#include "Functionalities.h"
#include "Precompute.h"
using namespace std;

extern bool LARGE_NETWORK;
extern Precompute PrecomputeObject;

Encoder::Encoder(EncoderConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->num_heads, conf->seq_len, conf->d_model, conf->batchSize, conf->d_ff, conf->num_layer)
{
  initialize();
};

void Encoder::initialize()
{
  encoders.resize(conf.num_layer);
  EncoderLayerConfig *cfg_encLayer = new EncoderLayerConfig(conf.num_heads, conf.seq_len, conf.d_model, conf.batchSize, conf.d_ff);
  for (int i = 0; i < conf.num_layer; ++i) {
    encoders[i] = new EncoderLayer(cfg_encLayer, 0);
  }
};

void Encoder::printLayer()
{
  cout << "----------------------------------------------" << endl;
  // cout << "(" << layerNum+1 << ") MHA Layer\t\t  " << conf.imageHeight << " x " << conf.imageWidth
  // 	 << " x " << conf.inputFeatures << endl << "\t\t\t  "
  // 	 << conf.filterSize << " x " << conf.filterSize << "  \t(Filter Size)" << endl << "\t\t\t  "
  // 	 << conf.stride << " , " << conf.padding << " \t(Stride, padding)" << endl << "\t\t\t  "
  // 	 << conf.batchSize << "\t\t(Batch Size)" << endl << "\t\t\t  "
  // 	 << (((conf.imageWidth - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x "
  // 	 << (((conf.imageHeight - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x "
  // 	 << conf.filters << " \t(Output)" << endl;
}

void Encoder::forward(const RSSVectorMyType &inputActivation)
{
  // Encoder(Q,K,V) =
  // RELUAttention(Q,K,V) = ReLU(QΩ)*RELU(KΩ)*V
  // q = inputActivation (batch_size, seq_len, d_model)
  // k = inputActivation (batch_size, seq_len, d_model)
  // v = inputActivation (batch_size, seq_len, d_model)
  log_print("Encoder.forward");

  encoders[0]->forward(inputActivation);
  for (int i = 1; i < conf.num_layer; ++i) {
    encoders[i]->forward(*(encoders[i-1]->getActivation()));
  }
}

// calc prevDelta
void Encoder::computeDelta(RSSVectorMyType &prevDelta)
{
  log_print("Encoder.computeDelta");

  for (int i = conf.num_layer-1; i >= 1; --i)
  {
    encoders[i]->computeDelta(*(encoders[i-1]->getDelta()));
  }
  encoders[0]->computeDelta(prevDelta);
}

void Encoder::updateEquations(const RSSVectorMyType &prevActivations)
{
  log_print("Encoder.updateEquations");

  for (int i = conf.num_layer-1; i >= 1; --i)
  {
    encoders[i]->updateEquations(*(encoders[i-1]->getActivation()));
  }
  encoders[0]->updateEquations(prevActivations);
}
