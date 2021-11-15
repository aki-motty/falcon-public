#include <iostream>
#include <string>
#include "AESObject.h"
#include "Precompute.h"
#include "secondary.h"
#include "connect.h"
#include "NeuralNetConfig.h"
#include "NeuralNetwork.h"
#include "unitTests.h"
#include "Functionalities.h"

#include "MHAttentionConfig.h"
#include "MHAttention.h"

int partyNum;
AESObject *aes_indep;
AESObject *aes_next;
AESObject *aes_prev;
Precompute PrecomputeObject;

int main(int argc, char **argv)
{
  /****************************** PREPROCESSING ******************************/
  parseInputs(argc, argv);
  NeuralNetConfig *config = new NeuralNetConfig(NUM_ITERATIONS);
  string network, dataset, security;
  bool PRELOADING = false;

  /****************************** SELECT NETWORK ******************************/
  // Network {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
  // Dataset {MNIST, CIFAR10, and ImageNet}
  // Security {Semi-honest or Malicious}
  if (argc == 9)
  {
    network = argv[6];
    dataset = argv[7];
    security = argv[8];
  }
  else
  {
    network = "SecureML";
    dataset = "MNIST";
    security = "Semi-honest";
  }
  selectNetwork(network, dataset, security, config);
  config->checkNetwork();
  NeuralNetwork *net = new NeuralNetwork(config);

  /****************************** AES SETUP and SYNC ******************************/
  aes_indep = new AESObject(argv[3]);
  aes_next = new AESObject(argv[4]);
  aes_prev = new AESObject(argv[5]);

  initializeCommunication(argv[2], partyNum);
  synchronize(2000000);

  /****************************** RUN NETWORK/UNIT TESTS ******************************/
  // Run these if you want a preloaded network to be tested
  // assert(NUM_ITERATION == 1 and "check if readMiniBatch is false in test(net)")
  // First argument {SecureML, Sarda, MiniONN, or LeNet}
  //  network += " preloaded"; PRELOADING = true;
  //  preload_network(PRELOADING, network, net);

  start_m();
  // Run unit tests in two modes:
  //	1. Debug {Mat-Mul, DotProd, PC, Wrap, ReLUPrime, ReLU, Division, BN, SSBits, SS, and Maxpool}
  //	2. Test {Mat-Mul1, Mat-Mul2, Mat-Mul3 (and similarly) Conv*, ReLU*, ReLUPrime*, and Maxpool*} where * = {1,2,3}
  //  runTest("Debug", "BN", network);
  //  runTest("Test", "ReLUPrime1", network);

  // Run forward/backward for single layers
  //  1. what {F, D, U} , F=forward, D=delta, U=update
  // 	2. l {0,1,....NUM_LAYERS-1}
  // size_t l = 0;
  // string what = "F";
  // runOnly(net, l, what, network);

  // Run training
  network += " train";
  // train(net);

  // Run inference (possibly with preloading a network)
  //  network += " test";
  //  test(PRELOADING, network, net);

  default_random_engine engine(0);
  uniform_real_distribution<> dist_real(0.0, 1.0);

  // test //////////////////////////
  size_t B = 1;
  size_t DM = 128;
  size_t H = 8;
  size_t L = 16;
  // size_t D = 4;
  size_t D = DM / H;
  // size_t M = 6;
  size_t M = static_cast<size_t>(D * log(D));
  bool causal = false;
  float attn_norm = 1.0;
  MHAttentionConfig *conf = new MHAttentionConfig(H, L, DM, B, causal, attn_norm, 0);
  MHAttention *layer = new MHAttention(conf, 0);

  size_t size = B * L * H * D;

  vector<myType> origin_input(size);
  RSSVectorMyType shared_input(size);
  for (size_t b = 0; b < B; ++b)
  {
    for (size_t l = 0; l < L; ++l)
    {
      for (size_t d = 0; d < DM; ++d)
      {
        origin_input[b * (L * DM) + l * (DM) + d] = floatToMyType((float)pow(1, b + l + d) * (b + l + d) * 0.1);
        // print_linear(origin_input[b * (L * DM) + l * (DM) + d], "FLOAT");
      }
      // cout << endl;
    }
    // cout << endl;
  }
  // cout << endl;
  funcGetShares(shared_input, origin_input);

  // set WQ, WK, WV, WO weights ///////////////////////////
  vector<myType> tmp_weights(DM * DM);
  for (size_t i = 0; i < DM; ++i)
  {
    for (size_t j = 0; j < DM; ++j)
    {
      tmp_weights[i * DM + j] = floatToMyType((float)(i + j) * 0.1);
    }
  }
  funcGetShares(*(layer->wq->getWeights()), tmp_weights);

  for (size_t i = 0; i < DM; ++i)
  {
    for (size_t j = 0; j < DM; ++j)
    {
      tmp_weights[i * DM + j] = floatToMyType((float)(i + j + 1) * 0.1);
    }
  }
  funcGetShares(*(layer->wk->getWeights()), tmp_weights);

  for (size_t i = 0; i < DM; ++i)
  {
    for (size_t j = 0; j < DM; ++j)
    {
      tmp_weights[i * DM + j] = floatToMyType((float)(i + j + 2) * 0.1);
    }
  }
  funcGetShares(*(layer->wv->getWeights()), tmp_weights);

  for (size_t i = 0; i < DM; ++i)
  {
    for (size_t j = 0; j < DM; ++j)
    {
      tmp_weights[i * DM + j] = floatToMyType((float)(i + j + 3) * 0.1);
    }
  }
  funcGetShares(*(layer->wo->getWeights()), tmp_weights);
  ////////////////////////////////////////////////////////////////////

  layer->forward(shared_input);

  end_m(network);
  // layer->wk->forward(shared_input);
  // layer->computeDelta(shared_input);

  // vector<myType> out(layer->getActivation()->size());
  // funcReconstruct(*(layer->getActivation()), out, out.size(), "out", true);
  // float sum = 0.0;
  // for (int i = 0; i < out.size(); ++i)
  // {
  //   sum += (static_cast<int64_t>(out[i])) / (float)(1 << FLOAT_PRECISION);
  // }
  // sum /= out.size();

  // cout << sum << endl;

  // vector<myType> tmp_delta(out.size(), floatToMyType(1.0 / out.size()));
  // funcGetShares(*(layer->getDelta()), tmp_delta);
  // RSSVectorMyType dX(shared_input.size());
  // // layer->wq->computeDelta(dX);
  // layer->computeDelta(dX);

  // vector<myType> out2(dX.size());
  // funcReconstruct(dX, out2, out2.size(), "out", true);
  // vector<myType> out2(layer->wq->getDelta()->size());
  // funcReconstruct(*(layer->wq->getDelta()), out2, out2.size(), "out", true);
  /*
    // size_t size = 100;
    vector<myType> origin_data(size);
    RSSVectorMyType shared_data(size);

    vector<myType> origin_proj(D * M);
    RSSVectorMyType shared_proj(D * M);

    vector<myType> origin_Q(B * L * H * M);
    RSSVectorMyType shared_Q(B * L * H * M);

    for (size_t b = 0; b < B; ++b) {
      for (size_t l = 0; l < L; ++l) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t d = 0; d < M; ++d) {
            origin_Q[b * (L * H * M) + l * (H * M) + h * M + d] = floatToMyType((float)(b + l + h + d));
            print_linear(origin_Q[b * (L * H * M) + l * (H * M) + h * M + d], "FLOAT");
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << endl;
    funcGetShares(shared_Q, origin_Q);

    vector<myType> origin_K(B * L * H * M);
    RSSVectorMyType shared_K(B * L * H * M);

    for (size_t b = 0; b < B; ++b) {
      for (size_t l = 0; l < L; ++l) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t d = 0; d < M; ++d) {
            origin_K[b * (L * H * M) + l * (H * M) + h * M + d] = floatToMyType((float)(b + l + h + d));
            print_linear(origin_K[b * (L * H * M) + l * (H * M) + h * M + d], "FLOAT");
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << endl;
    funcGetShares(shared_K, origin_K);

    vector<myType> origin_V(B * L * H * D);
    RSSVectorMyType shared_V(B * L * H * D);

    for (size_t b = 0; b < B; ++b) {
      for (size_t l = 0; l < L; ++l) {
        for (size_t h = 0; h < H; ++h) {
          for (size_t d = 0; d < D; ++d) {
            origin_V[b * (L * H * D) + l * (H * D) + h * D + d] = floatToMyType((float)(b + l + h + d));
            print_linear(origin_V[b * (L * H * D) + l * (H * D) + h * D + d], "FLOAT");
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << endl;
    }
    cout << endl;
    funcGetShares(shared_V, origin_V);

    RSSVectorMyType transQ(B*H*M*L);
      for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < L; ++j) {
          for (size_t k = 0; k < H; ++k) {
            for (size_t l = 0; l < M; ++l) {
              transQ[i * (H * L * M) + k * (L * M) + j * M + l] = shared_Q[i * (L * H * M) + j * (H * M) + k * M + l];
            }
          }
        }
      }

    // transpose K (B, L, H, M) -> (B, H, M, L)
      RSSVectorMyType transK(B*H*M*L);
      for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < L; ++j) {
          for (size_t k = 0; k < H; ++k) {
            for (size_t l = 0; l < M; ++l) {
              transK[i * (H * M * L) + k * (M * L) + l * L + j] = shared_K[i * (L * H * M) + j * (H * M) + k * M + l];
            }
          }
        }
      }

      // transpose V (B, L, H, D) -> (B, H, L, D)
      RSSVectorMyType transV(B*H*M*L);
      for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < L; ++j) {
          for (size_t k = 0; k < H; ++k) {
            for (size_t l = 0; l < D; ++l) {
              transV[i * (H * L * D) + k * (L * D) + j * D + l] = shared_V[i * (L * H * D) + j * (H * D) + k * D + l];
            }
          }
        }
      }

      // KV (B, H, M, L) * (B, H, L, D)-> (B, H, M, D)
      RSSVectorMyType KV(B*H*M*D);
      RSSVectorMyType inA(M * L);
      RSSVectorMyType inB(L * D);
      RSSVectorMyType out(M * D);
      for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < H; ++j) {
          copy(transK.begin() + (i * (H*M*L) + j * (M * L)), transK.begin() + (i * (H*M*L) + (j+1) * (M * L)), inA.begin());
          copy(transV.begin() + (i * (H*L*D) + j * (L * D)), transV.begin() + (i * (H*L*D) + (j+1) * (L * D)), inB.begin());
          funcMatMul(inA, inB, out, M, L, D, 0, 0, FLOAT_PRECISION);
          copy(out.begin(), out.end(), KV.begin() + i * (H*M*D) + j * (M*D));
        }
      }

      // QKV (B, H, L, M) * (B, H, M, D) -> (B, H, L, D)
      RSSVectorMyType QKV(B*L*H*D);
      RSSVectorMyType transQKV(B*H*L*D);
      inA.resize(L*M);
      inB.resize(M*D);
      out.resize(L*D);
      for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < H; ++j) {
          copy(transQ.begin() + (i * (H*L*M) + j * (L * M)), transQ.begin() + (i * (H*L*M) + (j+1) * (L * M)), inA.begin());
          copy(KV.begin() + (i * (H*M*D) + j * (M * D)), KV.begin() + (i * (H*M*D) + (j+1) * (M * D)), inB.begin());
          funcMatMul(inA, inB, out, L, M, D, 0, 0, FLOAT_PRECISION);
          copy(out.begin(), out.end(), transQKV.begin() + i * (H*L*D) + j * (L*D));
        }
      }

      // transpose QKV (B, H, L, D) -> (B, L, H, D)
      for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < H; ++j) {
          for (size_t k = 0; k < L; ++k) {
            for (size_t l = 0; l < D; ++l) {
              QKV[i * (L * H * D) + k * (H * D) + j * D + l] = transQKV[i * (H * L * D) + j * (L * D) + k * D + l];
            }
          }
        }
      }
  */
  // RSSVectorMyType QKV(B*L*H*D);
  // RSSVectorMyType sums(B*H*M*D);
  // for (size_t l = 0; l < L; ++l) {
  // 	for (size_t b = 0; b < B; ++b) {
  // 		for (size_t h = 0; h < H; ++h) {
  // 			RSSVectorMyType K_i(M);
  // 			copy(shared_K.begin() + (b * (L*H*M) + l * (H*M) + h * (M)), shared_K.begin() + (b * (L*H*M) + l * (H*M) + (h+1) * (M)), K_i.begin());
  // 			RSSVectorMyType V_i(D);
  // 			copy(shared_V.begin() + (b * (L*H*D) + l * (H*D) + h * (D)), shared_V.begin() + (b * (L*H*D) + l * (H*D) + (h+1) * (D)), V_i.begin());
  // 			RSSVectorMyType KV_i(M*D);
  // 			funcMatMul(K_i, V_i, KV_i, M, 1, D, 0, 0, FLOAT_PRECISION);
  // 			for (size_t t = 0; t < M*D; ++t) {
  // 				sums[b * (H*M*D) + h * (M*D) + t] = sums[b * (H*M*D) + h * (M*D) + t] + KV_i[t];
  // 			}

  // 			RSSVectorMyType Q_i(M);
  // 			copy(shared_Q.begin() + (b * (L*H*M) + l * (H*M) + h * (M)), shared_Q.begin() + (b * (L*H*M) + l * (H*M) + (h+1) * (M)), Q_i.begin());
  // 			RSSVectorMyType sum(M*D);
  // 			copy(sums.begin() + (b * (H*M*D) + h * (M*D)), sums.begin() + (b * (H*M*D) + (h+1) * (M*D)), sum.begin());
  // 			RSSVectorMyType QKV_i(D);
  // 			funcMatMul(Q_i, sum, QKV_i, 1, M, D, 0, 0, FLOAT_PRECISION);
  // 			copy(QKV_i.begin(), QKV_i.end(), QKV.begin() + b * (L*H*D) + l * (H*D) + h * D);
  // 		}
  // 	}
  // }

  // vector<myType> output(B*H*L*D);
  // funcReconstruct(QKV, output, output.size(), "KV", true);

  // for (size_t i = 0; i < size; ++i) {
  // 	data[i] = make_pair(floatToMyType(dist_real(engine)), floatToMyType(dist_real(engine)));
  // }

  // 	for (size_t b = 0; b < B; ++b) {
  // 		for (size_t l = 0; l < L; ++l) {
  // 			for (size_t h = 0; h < H; ++h) {
  // 				for (size_t d = 0; d < D; ++d) {
  // 					origin_data[b * (L * H * D) + l * (H * D) + h * D + d] = b + l + h + d;
  // 					cout << origin_data[b * (L * H * D) + l * (H * D) + h * D + d] << " ";
  // 				}
  // 				cout << endl;
  // 			}
  // 			cout << endl;
  // 		}
  // 		cout << endl;
  // 	}
  // 	cout << endl;
  //  cout << "#############################################" << endl;
  // 	for (size_t b = 0; b < B; ++b) {
  // 		for (size_t h = 0; h < H; ++h) {
  // 			for (size_t l = 0; l < L; ++l) {
  // 				for (size_t d = 0; d < D; ++d) {
  // 					cout << trans_data[b * (L * H * D) + h * (L * D) + l * D + d] << " ";
  // 				}
  // 				cout << endl;
  // 			}
  // 			cout << endl;
  // 		}
  // 		cout << endl;
  // 	}
  // 	cout << endl;

  // 	for (size_t d = 0; d < D; ++d) {
  // 		for (size_t f = 0; f < M; ++f) {
  // 			origin_proj[d * M + f] = d + f;
  // 			cout << origin_proj[d * M + f] << " ";
  // 		}
  // 		cout << endl;
  // 	}
  // 	cout << endl;

  // 	for (size_t i = 0; i < size; ++i) {
  // 		origin_data[i] = floatToMyType((float)(origin_data[i]) * 1);
  // 	}

  // 	for (size_t i = 0; i < D * M; ++i) {
  // 		origin_proj[i] = floatToMyType((float)(origin_proj[i]) * 1);
  // 	}

  // 	funcGetShares(shared_data, origin_data);
  // 	funcGetShares(shared_proj, origin_proj);

  // vector<myType> out(D * M);
  // funcReconstruct(shared_proj, out, D * M, "float", true);

  // cout << "output" << endl;
  // for (size_t b = 0; b < B; ++b) {
  // 	for (size_t l = 0; l < L; ++l) {
  // 		for (size_t h = 0; h < H; ++h) {
  // 			for (size_t d = 0; d < D; ++d) {
  // 				cout << (static_cast<int64_t>(out[b * (L * H * D) + l * (H * D) + h * D + d]))/(float)(1 << FLOAT_PRECISION) << " ";
  // 			}
  // 			cout << endl;
  // 		}
  // 		cout << endl;
  // 	}
  // 	cout << endl;
  // }
  // cout << endl;
  // vector<myType> origin_out(B * L * H * M);
  // RSSVectorMyType shared_out(B * L * H * M);
  // // test mhattentionLayer
  // layer->reluKernelTransformation(shared_data, shared_proj, M);
  // funcReconstruct(*(layer->relu1->getActivation()), origin_out, B * L * H * M, "out", true);

  // end_m(network);
  cout << "----------------------------------------------" << endl;
  cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum
       << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl
       << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
  cout << "----------------------------------------------" << endl
       << endl;

  // printNetwork(net);

  /****************************** CLEAN-UP ******************************/
  delete aes_indep;
  delete aes_next;
  delete aes_prev;
  delete config;
  delete net;
  deleteObjects();

  return 0;
}
