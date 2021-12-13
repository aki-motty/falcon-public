#pragma once
#include "Performer.h"
#include "Functionalities.h"
#include "Precompute.h"
using namespace std;

extern bool LARGE_NETWORK;
extern Precompute PrecomputeObject;

Performer::Performer(PerformerConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->num_heads, conf->seq_len, conf->d_model, conf->batchSize, conf->d_ff, conf->num_layer),
      encoder(encoder)
{
  initialize();
};

void Performer::initialize()
{
  EncoderConfig *cfg_encoder = new EncoderConfig(conf.num_heads, conf.seq_len, conf.d_model, conf.batchSize, conf.d_ff, conf.num_layer);
  encoder = new Encoder(cfg_encoder, 0);

  DecoderConfig *cfg_decoder = new DecoderConfig(conf.num_heads, conf.seq_len, conf.d_model, conf.batchSize, conf.d_ff, conf.num_layer);
  decoder = new Decoder(cfg_decoder, 0, encoder);

  FCConfig *cfg_fc = new FCConfig(conf.d_model, conf.batchSize * conf.seq_len, conf.d_model);
};

void Performer::printLayer()
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

void Performer::forward(const RSSVectorMyType &inputActivation)
{
  // Performer(Q,K,V) =
  // RELUAttention(Q,K,V) = ReLU(QΩ)*RELU(KΩ)*V
  // q = inputActivation (batch_size, seq_len, d_model)
  // k = inputActivation (batch_size, seq_len, d_model)
  // v = inputActivation (batch_size, seq_len, d_model)
  log_print("Performer.forward");

  encoder->forward(inputActivation);
  decoder->forward(*(encoder->getActivation()));

}

// calc prevDelta
void Performer::computeDelta(RSSVectorMyType &prevDelta)
{
  log_print("Performer.computeDelta");

  
}

void Performer::updateEquations(const RSSVectorMyType &prevActivations)
{
  log_print("Performer.updateEquations");

  
}
