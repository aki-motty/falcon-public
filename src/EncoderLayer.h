
#pragma once
#include "MHAttention.h"
#include "FFN.h"
#include "EncoderLayerConfig.h"
#include "LNLayer.h"
#include "FCLayer.h"
#include "ReLULayer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <random>
using namespace std;


class EncoderLayer : public Layer
{
private:
  RSSVectorMyType activations;
  RSSVectorMyType deltas;
  RSSVectorMyType res1;
  RSSVectorMyType res2;

  EncoderLayerConfig conf;
  MHAttention *mha;
  FFN *ffn;
  LNLayer *ln1;
  LNLayer *ln2;
  
public:
	
	//Constructor and initializer
	EncoderLayer(EncoderLayerConfig* conf, int _layerNum);
  ~EncoderLayer();
	void initialize();
 
	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;

	//Getters
	RSSVectorMyType* getActivation() {return &*(ln2->getActivation());};
	RSSVectorMyType* getDelta() {return &*(ln2->getDelta());};
};