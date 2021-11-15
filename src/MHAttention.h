
#pragma once
#include "MHAttentionConfig.h"
#include "FCLayer.h"
#include "ReLULayer.h"
#include "tools.h"
#include "connect.h"
#include "globals.h"
#include <random>
using namespace std;


class MHAttention : public Layer
{
private:
	MHAttentionConfig conf;
  // FCLayer* wq; // weights (d_model, d_model)
  
	
	RSSVectorMyType activations;
	RSSVectorMyType deltas;

	RSSVectorMyType constRatio;
	RSSVectorMyType constStab;
	RSSVectorMyType constAttenNorm;

	RSSVectorMyType projection_matrix;
	RSSVectorMyType KV;

	vector<float> createProjectionMatrix(size_t m, size_t d, size_t seed = 0, size_t scaling = 0);
	void createProductsOfGivensRotations(vector<float>& q, size_t dim, default_random_engine& engine);

public:
	FCLayer* wq; // weights (d_model, d_model)
	FCLayer* wk;
  FCLayer* wv;
  FCLayer* wo;
	ReLULayer* relu1;
	ReLULayer* relu2;
	
	//Constructor and initializer
	MHAttention(MHAttentionConfig* conf, int _layerNum);
  ~MHAttention();
	void initialize();
 
	//Functions
	void printLayer() override;
	void forward(const RSSVectorMyType& inputActivation) override;
	void computeDelta(RSSVectorMyType& prevDelta) override;
	void updateEquations(const RSSVectorMyType& prevActivations) override;

	//Getters
	RSSVectorMyType* getActivation() {return &*(wo->getActivation());};
	RSSVectorMyType* getDelta() {return &*(wo->getDelta());};
};