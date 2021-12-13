
#pragma once
#include "Performer.h"
#include "PerformerConfig.h"
#include "Encoder.h"
#include "Decoder.h"
#include "FCLayer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <random>
using namespace std;

class Performer : public Layer
{
private:
  PerformerConfig conf;
  Encoder *encoder;
  Decoder *decoder;
  FCLayer *fc;

public:
  // Constructor and initializer
  Performer(PerformerConfig *conf, int _layerNum);
  ~Performer();
  void initialize();

  // Functions
  void printLayer() override;
  void forward(const RSSVectorMyType &inputActivation) override;
  void computeDelta(RSSVectorMyType &prevDelta) override;
  void updateEquations(const RSSVectorMyType &prevActivations) override;

  // Getters
  // RSSVectorMyType *getActivation() { return &*((*(Performers.back())).getActivation()); };
  // RSSVectorMyType *getDelta() { return &*((*(Performers.back())).getDelta()); };
};