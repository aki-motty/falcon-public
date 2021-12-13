
#pragma once
#include "FFNConfig.h"
#include "FCLayer.h"
#include "ReLULayer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <random>
using namespace std;

class FFN : public Layer
{
private:
  FFNConfig conf;
  // FCLayer* wq; // weights (d_model, d_model)

  RSSVectorMyType activations;
  RSSVectorMyType deltas;

public:
  FCLayer *w1; // weights (d_model, d_model)
  FCLayer *w2; // weights (d_model, d_model)
  ReLULayer *relu;

  // Constructor and initializer
  FFN(FFNConfig *conf, int _layerNum);
  ~FFN();
  void initialize();

  // Functions
  void printLayer() override;
  void forward(const RSSVectorMyType &inputActivation) override;
  void computeDelta(RSSVectorMyType &prevDelta) override;
  void updateEquations(const RSSVectorMyType &prevActivations) override;

  // Getters
  RSSVectorMyType *getActivation() { return &*(w2->getActivation()); };
  RSSVectorMyType *getDelta() { return &*(w2->getDelta()); };
};