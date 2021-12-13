#pragma once
#include "Decoder.h"
#include "Functionalities.h"
#include "Precompute.h"
using namespace std;

extern bool LARGE_NETWORK;
extern Precompute PrecomputeObject;

Decoder::Decoder(DecoderConfig *conf, int _layerNum, Encoder *encoder)
    : Layer(_layerNum),
      conf(conf->num_heads, conf->seq_len, conf->d_model, conf->batchSize, conf->d_ff, conf->num_layer),
      encoder(encoder)
{
  initialize();
};

void Decoder::initialize()
{
  decoders.resize(conf.num_layer);
  DecoderLayerConfig *cfg_decLayer = new DecoderLayerConfig(conf.num_heads, conf.seq_len, conf.d_model, conf.batchSize, conf.d_ff);
  for (int i = 0; i < conf.num_layer; ++i)
  {
    decoders[i] = new DecoderLayer(cfg_decLayer, 0);
  }
};

void Decoder::printLayer()
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

void Decoder::forward(const RSSVectorMyType &inputActivation)
{
  // Decoder(Q,K,V) =
  // RELUAttention(Q,K,V) = ReLU(QΩ)*RELU(KΩ)*V
  // q = inputActivation (batch_size, seq_len, d_model)
  // k = inputActivation (batch_size, seq_len, d_model)
  // v = inputActivation (batch_size, seq_len, d_model)
  log_print("Decoder.forward");

  decoders[0]->forward(inputActivation, *(encoder->getActivation()));
  for (int i = 1; i < conf.num_layer; ++i)
  {
    decoders[i]->forward(*(decoders[i - 1]->getActivation()), *(encoder->getActivation()));
  }
}

// calc prevDelta
void Decoder::computeDelta(RSSVectorMyType &prevDelta)
{
  log_print("Decoder.computeDelta");

  for (int i = conf.num_layer - 1; i >= 1; --i)
  {
    decoders[i]->computeDelta(*(decoders[i - 1]->getDelta()), *(encoder->getDelta()));
  }
  decoders[0]->computeDelta(prevDelta, *(encoder->getDelta()));
}

void Decoder::updateEquations(const RSSVectorMyType &prevActivations)
{
  log_print("Decoder.updateEquations");

  for (int i = conf.num_layer - 1; i >= 1; --i)
  {
    decoders[i]->updateEquations(*(decoders[i - 1]->getActivation()), *(encoder->getActivation()));
  }
  decoders[0]->updateEquations(prevActivations, *(encoder->getActivation()));
}
