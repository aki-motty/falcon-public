
#pragma once
#include "DecoderLayer.h"
#include "DecoderConfig.h"
#include "Encoder.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <random>
using namespace std;

class Decoder : public Layer
{
private:
  DecoderConfig conf;
  vector<DecoderLayer *> decoders;
  Encoder *encoder;

public:
  // Constructor and initializer
  Decoder(DecoderConfig *conf, int _layerNum, Encoder *encoder);
  ~Decoder();
  void initialize();

  // Functions
  void printLayer() override;
  void forward(const RSSVectorMyType &inputActivation) override;
  void computeDelta(RSSVectorMyType &prevDelta) override;
  void updateEquations(const RSSVectorMyType &prevActivations) override;

  // Getters
  RSSVectorMyType *getActivation() { return &*((*(decoders.back())).getActivation()); };
  RSSVectorMyType *getDelta() { return &*((*(decoders.back())).getDelta()); };
};