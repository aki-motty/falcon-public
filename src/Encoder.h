
#pragma once
#include "EncoderLayer.h"
#include "EncoderConfig.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <random>
using namespace std;

class Encoder : public Layer
{
private:
  EncoderConfig conf;
  vector<EncoderLayer *> encoders;

public:
  // Constructor and initializer
  Encoder(EncoderConfig *conf, int _layerNum);
  ~Encoder();
  void initialize();

  // Functions
  void printLayer() override;
  void forward(const RSSVectorMyType &inputActivation) override;
  void computeDelta(RSSVectorMyType &prevDelta) override;
  void updateEquations(const RSSVectorMyType &prevActivations) override;

  // Getters
  RSSVectorMyType *getActivation() { return &*((*(encoders.back())).getActivation()); };
  RSSVectorMyType *getDelta() { return &*((*(encoders.back())).getDelta()); };
};