
#pragma once
#include "FFN.h"
#include "Functionalities.h"
using namespace std;

FFN::FFN(FFNConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->batchSize, conf->seq_len, conf->d_model, conf->d_ff)
{
  initialize();
}

void FFN::initialize()
{
  FCConfig *cfg_fc1 = new FCConfig(conf.d_model, conf.seq_len * conf.batchSize, conf.d_ff);
  FCConfig *cfg_fc2 = new FCConfig(conf.d_ff, conf.seq_len * conf.batchSize, conf.d_model);
  w1 = new FCLayer(cfg_fc1, 0); // 現在layernumを適当に0としている
  w2 = new FCLayer(cfg_fc2, 0); // 現在layernumを適当に0としている

  ReLUConfig *cfg_relu = new ReLUConfig(conf.d_ff, conf.seq_len * conf.batchSize);
  relu = new ReLULayer(cfg_relu, 0);
}

void FFN::printLayer()
{
  // cout << "----------------------------------------------" << endl;
  // cout << "(" << layerNum+1 << ") FC Layer\t\t  " << conf.inputDim << " x " << conf.outputDim << endl << "\t\t\t  "
  // 	 << conf.batchSize << "\t\t (Batch Size)" << endl;
}

void FFN::forward(const RSSVectorMyType &inputActivation)
{
  log_print("FC.forward");

  size_t B = conf.batchSize;
  size_t SL = conf.seq_len;
  size_t DM = conf.d_model;
  size_t DFF = conf.d_ff;

  w1->forward(inputActivation);
  relu->forward(*w1->getActivation());
  w2->forward(*relu->getActivation());
}

void FFN::computeDelta(RSSVectorMyType &prevDelta)
{
  log_print("FC.computeDelta");

  w2->computeDelta(*(relu->getDelta()));
  relu->computeDelta(*(w1->getDelta()));
  w1->computeDelta(prevDelta);
}

void FFN::updateEquations(const RSSVectorMyType &prevActivations)
{
  log_print("FC.updateEquations");

  w2->updateEquations(*(relu->getActivation()));
  w1->updateEquations(prevActivations);
}
