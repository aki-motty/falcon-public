#pragma once
#include "EncoderLayer.h"
#include "Functionalities.h"
#include "Precompute.h"
using namespace std;

extern bool LARGE_NETWORK;
extern Precompute PrecomputeObject;

EncoderLayer::EncoderLayer(EncoderLayerConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->num_heads, conf->seq_len, conf->d_model, conf->batchSize, conf->d_ff),
      activations(conf->batchSize * conf->seq_len * conf->d_model),
      deltas(conf->batchSize * conf->seq_len * conf->d_model),
      res1(conf->batchSize * conf->seq_len * conf->d_model),
      res2(conf->batchSize * conf->seq_len * conf->d_model)
{
  initialize();
};

void EncoderLayer::initialize()
{
  MHAttentionConfig *cfg_mha = new MHAttentionConfig(conf.num_heads, conf.seq_len, conf.d_model, conf.batchSize, false, 0.01, 0);
  mha = new MHAttention(cfg_mha, 0);

  FFNConfig *cfg_ffn = new FFNConfig(conf.batchSize, conf.seq_len, conf.d_model, conf.d_ff);
  ffn = new FFN(cfg_ffn, 0);

  LNConfig *cfg_ln = new LNConfig(conf.d_model, conf.batchSize * conf.seq_len);
  ln1 = new LNLayer(cfg_ln, 0);
  ln2 = new LNLayer(cfg_ln, 0);


};

void EncoderLayer::printLayer()
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

void EncoderLayer::forward(const RSSVectorMyType &inputActivation)
{
  // EncoderLayer(Q,K,V) =
  // RELUAttention(Q,K,V) = ReLU(QΩ)*RELU(KΩ)*V
  // q = inputActivation (batch_size, seq_len, d_model)
  // k = inputActivation (batch_size, seq_len, d_model)
  // v = inputActivation (batch_size, seq_len, d_model)
  log_print("EncoderLayer.forward");

  size_t B = conf.batchSize;
  size_t DM = conf.d_model;
  size_t SL = conf.seq_len;

  mha->forward(inputActivation);
  addVectors<RSSMyType>(inputActivation, *(mha->getActivation()), res1, B*SL*DM);
  ln1->forward(res1);

  ffn->forward(*(ln1->getActivation()));
  addVectors<RSSMyType>(*(ln1->getActivation()), *(ffn->getActivation()), res2, B * SL * DM);
  ln2->forward(res2);
}

// calc prevDelta
void EncoderLayer::computeDelta(RSSVectorMyType &prevDelta)
{
  log_print("EncoderLayer.computeDelta");

  size_t size = conf.batchSize * conf.seq_len * conf.d_model;

  ln2->computeDelta(*(ffn->getDelta()));
  RSSVectorMyType d_ffn(size);
  ffn->computeDelta(d_ffn);
  addVectors<RSSMyType>(*(ffn->getDelta()), d_ffn, *(ln1->getDelta()), size);

  ln1->computeDelta(*(mha->getDelta()));
  RSSVectorMyType dmha(size);
  mha->computeDelta(dmha);
  addVectors<RSSMyType>(*(mha->getDelta()), dmha, prevDelta, size);
}

void EncoderLayer::updateEquations(const RSSVectorMyType &prevActivations)
{
  log_print("EncoderLayer.updateEquations");

  ln2->updateEquations(res2);
  ffn->updateEquations(*(ln1->getActivation()));
  ln1->updateEquations(res1);
  mha->updateEquations(prevActivations);
}
