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
#include "EncoderLayerConfig.h"
#include "EncoderLayer.h"
#include "DecoderLayerConfig.h"
#include "DecoderLayer.h"
#include "EncoderConfig.h"
#include "Encoder.h"
#include "DecoderConfig.h"
#include "Decoder.h"
#include "FFNConfig.h"
#include "FFN.h"
#include "LNLayer.h"
#include "LNConfig.h"

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
  size_t DM = 512;
  size_t H = 8;
  size_t L =  8;
  size_t DFF = 2048;
  size_t num_layer = 6;
  // size_t B = 1;
  // size_t DM = 8;
  // size_t H = 2;
  // size_t L = 8;
  // size_t DFF = 4;
  // size_t num_layer = 2;
  size_t D = DM / H;
  size_t M = static_cast<size_t>(D * log(D));
  bool causal = true;
  float attn_norm = 1.0;

  cout << "B : " << B << endl;
  cout << "DM : " << DM << endl;
  cout << "H : " << H << endl;
  cout << "L : " << L << endl;
  cout << "DFF : " << DFF << endl;
  cout << "num_layer : " << num_layer << endl;
  cout << "D : " << D << endl;
  cout << "M : " << M << endl;
  cout << "causal : " << causal << endl;
 
  MHAttentionConfig *cfg_mha = new MHAttentionConfig(H, L, DM, B, causal, attn_norm, 0);
  MHAttention *mha = new MHAttention(cfg_mha, 0);
  // EncoderLayerConfig *cfg_encl = new EncoderLayerConfig(H, L, DM, B, DFF);
  // EncoderLayer *encl = new EncoderLayer(cfg_encl, 0);
  // EncoderConfig *cfg_enc = new EncoderConfig(H, L, DM, B, DFF, num_layer);
  // Encoder *enc = new Encoder(cfg_enc, 0);
  // DecoderLayerConfig *cfg_decl = new DecoderLayerConfig(H, L, DM, B, DFF);
  // DecoderLayer *decl = new DecoderLayer(cfg_decl, 0);
  // DecoderConfig *cfg_dec = new DecoderConfig(H, L, DM, B, DFF, num_layer);
  // Decoder *dec = new Decoder(cfg_dec, 0, enc);
  // FFNConfig *cfg_ffn = new FFNConfig(B, L, DM, DFF);
  // FFN *ffn = new FFN(cfg_ffn, 0);
  // LNConfig *cfg_ln = new LNConfig(DM, B * L);
  // LNLayer *ln = new LNLayer(cfg_ln, 0);

  size_t size = B * L * H * D;

  cout << size << endl;

  vector<myType> origin_input(size);
  RSSVectorMyType shared_input(size);
  for (size_t b = 0; b < B; ++b)
  {
    for (size_t l = 0; l < L; ++l)
    {
      for (size_t d = 0; d < DM; ++d)
      {
        origin_input[b * (L * DM) + l * (DM) + d] = floatToMyType((float)pow(1, b + l + d) * (b + l + d + 0.01) * 0.1);
        // print_linear(origin_input[b * (L * DM) + l * (DM) + d], "FLOAT");
      }
      // cout << endl;
    }
    // cout << endl;
  }
  // cout << endl;
  funcGetShares(shared_input, origin_input);
  // start_m();
  for (int i = 0; i < 5; ++i) {
      mha->forward(shared_input);
      // ffn->forward(shared_input);
      // ln->forward(shared_input);
      // encl->forward(shared_input);
      // enc->forward(shared_input);
      // decl->forward(shared_input, shared_input);
      // dec->forward(shared_input);
  }

  // end_m("forward");

  // RSSVectorMyType prevDelta(size);
  // RSSVectorMyType prevEncoderDelta(size);
  // start_m();
  // mha->computeDelta(prevDelta);
  // ffn->computeDelta(prevDelta);
  // ln->computeDelta(prevDelta);
  // encl->computeDelta(prevDelta);
  // enc->computeDelta(prevDelta);
  // decl->computeDelta(prevDelta, prevEncoderDelta);
  // dec->computeDelta(prevDelta);

  // mha->updateEquations(shared_input);
  // ffn->updateEquations(shared_input);
  // ln->updateEquations(shared_input);
  // encl->updateEquations(shared_input);
  // enc->updateEquations(shared_input);
  // decl->updateEquations(shared_input, shared_input);
  // dec->updateEquations(shared_input);
  // end_m("mhabackward");
  // vector<myType> tmp(size);
  // RSSVectorMyType alpha(size);
  // RSSVectorMyType beta(size);
  // funcApproxInverseSqrt(shared_input, alpha, beta, size);
  // funcReconstruct(alpha, tmp, size, "alpha", true);
  // funcReconstruct(beta, tmp, size, "beta", true);

  // vector<myType> reverse_input(size);
  // RSSVectorMyType quotient(size);
  // copyVectors(origin_input, reverse_input, size);
  // reverse(reverse_input.begin(), reverse_input.end());
  // RSSVectorMyType shared_reverse_input(size);
  // funcGetShares(shared_reverse_input, reverse_input);
  // for (int i = 0; i < size; ++i) print_linear(reverse_input[i], "FLOAT");
  // cout << endl;
  // for (int i = 0; i < size; ++i) cout << origin_input[i] / (double)reverse_input[i] << " ";
  // cout << endl;
  // funcDivision2(shared_input, shared_reverse_input, quotient, size);
  // funcReconstruct(quotient, tmp, size, "quotient", true);

  // vector<myType> origin_QK(B*H*L*L);
  // RSSVectorMyType shared_QK(B*H*L*L);
  // for (size_t b = 0; b < B; ++b)
  // {
  //   for (size_t h = 0; h < H; ++h) {
  //     for (size_t l = 0; l < L; ++l) {
  //       for (size_t k = 0; k < L; ++k) {
  //         origin_QK[b * (H * L * L) + h * (L * L) + l * L + k] = floatToMyType((float)pow(1, b + l + h + k) * (b + l + h + k + 0.01) * 0.1);
  //         // print_linear(origin_QK[b * (H * L * L) + h * (L * L) + l * L + k], "FLOAT");
  //       }
  //       // cout << endl;
  //     }
  //     // cout << endl;
  //   }
  //   // cout << endl;
  // }
  // // cout << endl;
  // funcGetShares(shared_QK, origin_QK);

  // vector<myType> tmp(B*H*L*L);
  // RSSVectorMyType out(B*H*L*L);
  // start_m();
  // for (int i = 0; i < 10; ++i) {
  //   funcSoftmax(shared_QK, out, B*H, L, L, true);
  // }
  // end_m("softmax");
  
  // funcReconstruct(out, tmp, B*H*L*L, "softmax", true);
  // for (size_t b = 0; b < B; ++b)
  // {
  //   for (size_t h = 0; h < H; ++h) {
  //     for (size_t l = 0; l < L; ++l) {
  //       for (size_t k = 0; k < L; ++k) {
  //         print_linear(tmp[b * (H * L * L) + h * (L * L) + l * L + k], "FLOAT");
  //       }
  //       cout << endl;
  //     }
  //     cout << endl;
  //   }
  //   cout << endl;
  // }
  // cout << endl;

  // vector<myType> tmp(B*H*L*D);
  // RSSVectorMyType out(B*H*L*D);
  // start_m();
  // for (int i = 0; i < 10; ++i) {
    
  //     funcSoftmaxAttetion(shared_input, shared_input, shared_input, out, B, L, H, D, causal);
    
  // }
  // end_m("softmaxattention");
  
  // funcReconstruct(out, tmp, B*H*L*D, "softmaxAttention", true);

  // // set WQ, WK, WV, WO weights ///////////////////////////
  // vector<myType> tmp_weights(DM * DM);
  // for (size_t i = 0; i < DM; ++i)
  // {
  //   for (size_t j = 0; j < DM; ++j)
  //   {
  //     tmp_weights[i * DM + j] = floatToMyType((float)(i + j) * 0.1);
  //   }
  // }
  // funcGetShares(*(layer->wq->getWeights()), tmp_weights);

  // for (size_t i = 0; i < DM; ++i)
  // {
  //   for (size_t j = 0; j < DM; ++j)
  //   {
  //     tmp_weights[i * DM + j] = floatToMyType((float)(i + j + 1) * 0.1);
  //   }
  // }
  // funcGetShares(*(layer->wk->getWeights()), tmp_weights);

  // for (size_t i = 0; i < DM; ++i)
  // {
  //   for (size_t j = 0; j < DM; ++j)
  //   {
  //     tmp_weights[i * DM + j] = floatToMyType((float)(i + j + 2) * 0.1);
  //   }
  // }
  // funcGetShares(*(layer->wv->getWeights()), tmp_weights);

  // for (size_t i = 0; i < DM; ++i)
  // {
  //   for (size_t j = 0; j < DM; ++j)
  //   {
  //     tmp_weights[i * DM + j] = floatToMyType((float)(i + j + 3) * 0.1);
  //   }
  // }
  // funcGetShares(*(layer->wo->getWeights()), tmp_weights);
  ////////////////////////////////////////////////////////////////////

  // layer->forward(shared_input);

  // vector<myType> origin_delta(size);
  // RSSVectorMyType shared_delta(size);
  // for (size_t b = 0; b < B; ++b)
  // {
  //     for (size_t l = 0; l < L; ++l)
  //     {
  //         for (size_t d = 0; d < DM; ++d)
  //         {
  //             origin_delta[b * (L * DM) + l * (DM) + d] = floatToMyType((float)1.);
  //         }
  //     }
  // }
  // funcGetShares(*layer->getDelta(), origin_delta);
  // layer->computeDelta(shared_delta);

  // layer->updateEquations(shared_input);

  // end_m(network);
  // cout << "----------------------------------------------" << endl;
  // cout << "Run details: " << NUM_OF_PARTIES << "PC (P" << partyNum
  //      << "), " << NUM_ITERATIONS << " iterations, batch size " << MINI_BATCH_SIZE << endl
  //      << "Running " << security << " " << network << " on " << dataset << " dataset" << endl;
  // cout << "----------------------------------------------" << endl
  //      << endl;

  /****************************** CLEAN-UP ******************************/
  delete aes_indep;
  delete aes_next;
  delete aes_prev;
  delete config;
  delete net;
  deleteObjects();

  return 0;
}
