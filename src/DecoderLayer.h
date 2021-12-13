
#pragma once
#include "MHAttention.h"
#include "FFN.h"
#include "DecoderLayerConfig.h"
#include "LNLayer.h"
#include "FCLayer.h"
#include "ReLULayer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <random>
using namespace std;

class DecoderLayer : public Layer
{
private:
  RSSVectorMyType activations;
  RSSVectorMyType deltas;
  RSSVectorMyType res1;
  RSSVectorMyType res2;
  RSSVectorMyType res3;

  DecoderLayerConfig conf;
  MHAttention *mha;
  MHAttention *masked_mha;
  FFN *ffn;
  LNLayer *ln1;
  LNLayer *ln2;
  LNLayer *ln3;

public:
  // Constructor and initializer
  DecoderLayer(DecoderLayerConfig *conf, int _layerNum);
  ~DecoderLayer();
  void initialize();

  // Functions
  void printLayer() override;
  void forward(const RSSVectorMyType &inputActivation) override;
  void forward(const RSSVectorMyType &inputActivation, const RSSVectorMyType &encoderActivation);
  void computeDelta(RSSVectorMyType &prevDelta) override;
  void computeDelta(RSSVectorMyType &prevDelta, RSSVectorMyType &encoderDelta);
  void updateEquations(const RSSVectorMyType &prevActivations) override;
  void updateEquations(const RSSVectorMyType &prevActivations, const RSSVectorMyType &encoderActivations);

  // Getters
  RSSVectorMyType *getActivation() { return &*(ln3->getActivation()); };
  RSSVectorMyType *getDelta() { return &*(ln3->getDelta()); };
};