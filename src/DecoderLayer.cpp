#pragma once
#include "DecoderLayer.h"
#include "Functionalities.h"
#include "Precompute.h"
using namespace std;

extern bool LARGE_NETWORK;
extern Precompute PrecomputeObject;

DecoderLayer::DecoderLayer(DecoderLayerConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->num_heads, conf->seq_len, conf->d_model, conf->batchSize, conf->d_ff),
      activations(conf->batchSize * conf->seq_len * conf->d_model),
      deltas(conf->batchSize * conf->seq_len * conf->d_model),
      res1(conf->batchSize * conf->seq_len * conf->d_model),
      res2(conf->batchSize * conf->seq_len * conf->d_model),
      res3(conf->batchSize * conf->seq_len * conf->d_model)
{
  initialize();
};

void DecoderLayer::initialize()
{
  MHAttentionConfig *cfg_mha = new MHAttentionConfig(conf.num_heads, conf.seq_len, conf.d_model, conf.batchSize, false, 0.01, 0);
  mha = new MHAttention(cfg_mha, 0);
  MHAttentionConfig *cfg_masked_mha = new MHAttentionConfig(conf.num_heads, conf.seq_len, conf.d_model, conf.batchSize, true, 0.01, 0);
  masked_mha = new MHAttention(cfg_masked_mha, 0);

  FFNConfig *cfg_ffn = new FFNConfig(conf.batchSize, conf.seq_len, conf.d_model, conf.d_ff);
  ffn = new FFN(cfg_ffn, 0);

  LNConfig *cfg_ln = new LNConfig(conf.d_model, conf.batchSize * conf.seq_len);
  ln1 = new LNLayer(cfg_ln, 0);
  ln2 = new LNLayer(cfg_ln, 0);
  ln3 = new LNLayer(cfg_ln, 0);
};

void DecoderLayer::printLayer()
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

void DecoderLayer::forward(const RSSVectorMyType &inputActivation)
{
  // DecoderLayer(Q,K,V) =
  // RELUAttention(Q,K,V) = ReLU(QΩ)*RELU(KΩ)*V
  // q = inputActivation (batch_size, seq_len, d_model)
  // k = inputActivation (batch_size, seq_len, d_model)
  // v = inputActivation (batch_size, seq_len, d_model)
  log_print("DecoderLayer.forward");

  cerr << "Not implemented." << endl;
  exit(EXIT_FAILURE);
}

void DecoderLayer::forward(const RSSVectorMyType &inputActivation, const RSSVectorMyType &encoderActivation)
{
  // DecoderLayer(Q,K,V) =
  // RELUAttention(Q,K,V) = ReLU(QΩ)*RELU(KΩ)*V
  // q = inputActivation (batch_size, seq_len, d_model)
  // k = inputActivation (batch_size, seq_len, d_model)
  // v = inputActivation (batch_size, seq_len, d_model)
  log_print("DecoderLayer.forward");

  size_t B = conf.batchSize;
  size_t DM = conf.d_model;
  size_t SL = conf.seq_len;

  masked_mha->forward(inputActivation);
  addVectors<RSSMyType>(inputActivation, *(masked_mha->getActivation()), res1, B * SL * DM);
  ln1->forward(res1);

  mha->forward(encoderActivation, *(ln1->getActivation()));
  addVectors<RSSMyType>(*(ln1->getActivation()), *(mha->getActivation()), res2, B * SL * DM);
  ln2->forward(res2);

  ffn->forward(*(ln2->getActivation()));
  addVectors<RSSMyType>(*(ln2->getActivation()), *(ffn->getActivation()), res3, B * SL * DM);
  ln3->forward(res3);
}

// calc prevDelta
void DecoderLayer::computeDelta(RSSVectorMyType &prevDelta)
{
  log_print("DecoderLayer.computeDelta");

  cerr << "Not implemented." << endl;
  exit(EXIT_FAILURE);
}

void DecoderLayer::computeDelta(RSSVectorMyType &prevDelta, RSSVectorMyType &encoderDelta)
{
  log_print("DecoderLayer.computeDelta");

  size_t size = conf.batchSize * conf.seq_len * conf.d_model;

  ln3->computeDelta(*(ffn->getDelta()));
  RSSVectorMyType d_ffn(size);
  ffn->computeDelta(d_ffn);
  addVectors<RSSMyType>(*(ffn->getDelta()), d_ffn, *(ln2->getDelta()), size);

  ln2->computeDelta(*(mha->getDelta()));
  RSSVectorMyType d_encoder_mha(size);
  RSSVectorMyType d_decoder_mha(size);
  mha->computeDelta(d_encoder_mha,d_decoder_mha);
  addVectors<RSSMyType>(*(mha->getDelta()), d_decoder_mha, *(ln1->getDelta()), size);
  addVectors<RSSMyType>(d_encoder_mha, encoderDelta, encoderDelta, size);

  ln1->computeDelta(*(masked_mha->getDelta()));
  RSSVectorMyType d_masked_mha(size);
  masked_mha->computeDelta(d_masked_mha);
  addVectors<RSSMyType>(*(masked_mha->getDelta()), d_masked_mha, prevDelta, size);
}

void DecoderLayer::updateEquations(const RSSVectorMyType &prevActivations)
{
  log_print("DecoderLayer.updateEquations");

  cerr << "Not implemented." << endl;
  exit(EXIT_FAILURE);
}

void DecoderLayer::updateEquations(const RSSVectorMyType &prevActivations, const RSSVectorMyType &encoderActivations)
{
  log_print("DecoderLayer.updateEquations");

  ln3->updateEquations(res3);
  ffn->updateEquations(*(ln2->getActivation()));
  ln2->updateEquations(res2);
  mha->updateEquations(encoderActivations, *(ln1->getActivation()));
  ln1->updateEquations(res1);
  masked_mha->updateEquations(prevActivations);
}
