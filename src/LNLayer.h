
#pragma once
#include "LNConfig.h"
#include "Layer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
using namespace std;

class LNLayer : public Layer
{
private:
  LNConfig conf;
  RSSVectorMyType activations;
  RSSVectorMyType deltas;
  RSSVectorMyType gamma;
  RSSVectorMyType beta;
  RSSVectorMyType xhat;
  RSSVectorMyType sigma;

public:
  // Constructor and initializer
  LNLayer(LNConfig *conf, int _layerNum);
  void initialize();

  // Functions
  void printLayer() override;
  void forward(const RSSVectorMyType &inputActivation) override;
  void computeDelta(RSSVectorMyType &prevDelta) override;
  void updateEquations(const RSSVectorMyType &prevActivations) override;

  // Getters
  RSSVectorMyType *getActivation() { return &activations; };
  RSSVectorMyType *getDelta() { return &deltas; };
};