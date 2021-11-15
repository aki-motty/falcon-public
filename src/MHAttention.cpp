#pragma once
#include "MHAttention.h"
#include "Functionalities.h"
#include "Precompute.h"
using namespace std;

extern bool LARGE_NETWORK;
extern Precompute PrecomputeObject;

MHAttention::MHAttention(MHAttentionConfig *conf, int _layerNum)
    : Layer(_layerNum),
      conf(conf->num_heads, conf->seq_len, conf->d_model, conf->batchSize, conf->causal, conf->attention_normalizer, conf->seed),
      activations(conf->batchSize * conf->seq_len * conf->d_model),
      deltas(conf->batchSize * conf->seq_len * conf->d_model),
      constRatio(conf->batchSize * conf->seq_len * conf->num_heads * conf->nb_features),
      constStab(conf->batchSize * conf->seq_len * conf->num_heads * conf->nb_features),
      constAttenNorm(conf->batchSize * conf->seq_len * conf->num_heads * conf->depth),
      projection_matrix(conf->nb_features * conf->depth),
      KV(conf->batchSize * conf->num_heads * conf->nb_features * conf->seq_len)
{
  initialize();
};

void MHAttention::initialize()
{
  FCConfig *cfg_fc = new FCConfig(conf.d_model, conf.seq_len * conf.batchSize, conf.d_model);
  wq = new FCLayer(cfg_fc, 0); // 現在layernumを適当に0としている
  wk = new FCLayer(cfg_fc, 0); // 現在layernumを適当に0としている
  wv = new FCLayer(cfg_fc, 0); // 現在layernumを適当に0としている
  wo = new FCLayer(cfg_fc, 0); // 現在layernumを適当に0としている

  ReLUConfig *cfg_relu = new ReLUConfig(conf.nb_features, conf.batchSize * conf.seq_len * conf.num_heads);
  relu1 = new ReLULayer(cfg_relu, 0);
  relu2 = new ReLULayer(cfg_relu, 0);

  const myType tmp_ratio = floatToMyType((float)(1.0 / sqrt(conf.nb_features)));
  // print_myType(constRatio, "const Ratio", "FLOAT");
  vector<myType> ratio(constRatio.size(), tmp_ratio);
  funcGetShares(constRatio, ratio);

  const myType tmp_stab = 1;
  // print_myType(constStab, "const stab", "FLOAT");
  vector<myType> stab(constStab.size(), tmp_stab);
  funcGetShares(constStab, stab);

  const myType tmp_attenNorm = floatToMyType(conf.attention_normalizer);
  print_myType(tmp_attenNorm, "const attention norm", "FLOAT");
  vector<myType> attenNorm(constAttenNorm.size(), tmp_attenNorm);
  funcGetShares(constAttenNorm, attenNorm);
};

void MHAttention::printLayer()
{
  cout << "----------------------------------------------" << endl;
  // cout << "(" << layerNum+1 << ") MHA Layer\t\t  " << conf.imageHeight << " x " << conf.imageWidth
  // 	 << " x " << conf.inputFeatures << endl << "\t\t\t  "
  // 	 << conf.filterSize << " x " << conf.filterSize << "  \t(Filter Size)" << endl << "\t\t\t  "
  // 	 << conf.stride << " , " << conf.padding << " \t(Stride, padding)" << endl << "\t\t\t  "
  // 	 << conf.batchSize << "\t\t(Batch Size)" << endl << "\t\t\t  "
  // 	 << (((conf.imageWidth - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x "
  // 	 << (((conf.imageHeight - conf.filterSize + 2*conf.padding)/conf.stride) + 1) << " x "
  // 	 << conf.filters << " \t(Output)" << endl;
}

vector<float> MHAttention::createProjectionMatrix(size_t m, size_t D, size_t seed, size_t scaling)
{
  /* Constructs the matrix of random projections.
  Args:
    m: number of random projections.
    D: dimensionality of eatch random projection.
    seed: random seed used to construct projections.
    scaling: 1 if all the random projections need to be renormalized to have
      length \sqrt{D}, 0 if the lengths of random projections should follow
      \chi(D) distribution.
    struct_mode: if True then products of Givens rotations will be used to
      construct random orthogonal matrix. This bypasses Gram-Schmidt orthogonalization.
  Returns:
    The matrix of random projections of the shape (m, D).
  */

  assert((scaling == 0 || scaling == 1) && "Scaling must be one of {0, 1}.");

  size_t nb_full_blocks = m / D;
  default_random_engine engine(seed);

  vector<float> q(m * D);
  vector<float> tmp(D * D);
  for (size_t i = 0; i < nb_full_blocks; ++i)
  {
    createProductsOfGivensRotations(tmp, D, engine);
    copy(tmp.begin(), tmp.end(), q.begin() + (i * (D * D)));
  }
  size_t remaining_rows = m - nb_full_blocks * D;
  if (remaining_rows > 0)
  {
    createProductsOfGivensRotations(tmp, D, engine);
    copy(tmp.begin(), tmp.begin() + (remaining_rows * D), q.begin() + (nb_full_blocks * D * D));
  }

  normal_distribution<> dist_normal(0.0, 1.0);
  if (scaling == 0)
  {
    for (long i = 0; i < m; ++i)
    {
      // calc norm
      float sum = 0.0;
      for (long j = 0; j < D; ++j)
      {
        float t = dist_normal(engine);
        sum += t * t;
      }
      float multiplier = sqrt(sum);
      for (long j = 0; j < D; ++j)
      {
        q[i * D + j] *= multiplier;
      }
    }
  }
  else
  {
    float multiplier = sqrt(D);
    for (long i = 0; i < m; ++i)
    {
      for (long j = 0; j < D; ++j)
      {
        q[i * D + j] *= multiplier;
      }
    }
  }
  return q;
}

void MHAttention::createProductsOfGivensRotations(vector<float> &q, size_t dim, default_random_engine &engine)
{
  /* Constructs a 2D-tensor which is a product of Givens random rotation.
  Args:
    dim: number of rows/columns of the resulting 2D-tensor.
    seed: random seed.
  Returns:
    The product of Givens random rotations.
  */

  uniform_real_distribution<> dist_real(0.0, 1.0);
  uniform_int_distribution<> dist_int(0, dim - 1);
  size_t nb_givens_rotations = dim * static_cast<size_t>(ceil(log(static_cast<float>(dim))));
  for (long i = 0; i < dim; ++i)
  {
    for (long j = 0; j < dim; ++j)
    {
      if (i == j)
      {
        q[i * dim + j] = 1.0;
      }
      else
      {
        q[i * dim + j] = 0.0;
      }
    }
  }
  for (size_t _ = 0; _ < nb_givens_rotations; ++_)
  {
    float random_angle = M_PI * dist_real(engine);
    size_t random_indice1 = dist_int(engine);
    size_t random_indice2 = dist_int(engine);
    size_t index_i = min(random_indice1, random_indice2);
    size_t index_j = max(random_indice1, random_indice2);
    vector<float> tmp_i(dim);
    vector<float> tmp_j(dim);
    for (size_t D = 0; D < dim; ++D)
    {
      tmp_i[D] = cos(random_angle) * q[index_i * dim + D] + sin(random_angle) * q[index_j * dim + D];
      tmp_j[D] = -sin(random_angle) * q[index_i * dim + D] + cos(random_angle) * q[index_j * dim + D];
    }

    for (size_t D = 0; D < dim; ++D)
    {
      q[index_i * dim + D] = tmp_i[D];
      q[index_j * dim + D] = tmp_j[D];
    }
  }
}

void MHAttention::forward(const RSSVectorMyType &inputActivation)
{
  // MHAttention(Q,K,V) =
  // RELUAttention(Q,K,V) = ReLU(QΩ)*RELU(KΩ)*V
  // q = inputActivation (batch_size, seq_len, d_model)
  // k = inputActivation (batch_size, seq_len, d_model)
  // v = inputActivation (batch_size, seq_len, d_model)
  log_print("MHAttention.forward");

  size_t B = conf.batchSize;
  size_t NH = conf.num_heads;
  size_t DM = conf.d_model;
  size_t NF = conf.nb_features;
  size_t SL = conf.seq_len;
  size_t D = conf.depth;
  float attention_normalizer = conf.attention_normalizer;

  // consider (batch_size, seq_len, d_model) as (batch_size * seq_len, d_model)
  wq->forward(inputActivation); // (batch_size * seq_len, d_model)
  wk->forward(inputActivation); // (batch_size * seq_len, d_model)
  wv->forward(inputActivation); // (batch_size * seq_len, d_model)

  // create projection matrix
  vector<float> proj_float = createProjectionMatrix(NF, D, conf.seed, 1);
  vector<myType> proj_myType(proj_float.size());
  // for (size_t i = 0; i < proj_myType.size(); ++i) {
  // 	proj_myType[i] = floatToMyType(proj_float[i]);
  // }
  for (size_t i = 0; i < NF; ++i)
  {
    for (size_t j = 0; j < D; ++j)
    {
      proj_myType[i * D + j] = floatToMyType((float)(i + j));
      // print_linear(proj_myType[i * D + j], "FLOAT");
    }
    // cout << endl;
  }
  // cout << endl;
  funcGetShares(projection_matrix, proj_myType);

  // consider (batch_size * seq_len, d_model) as (batch_size * seq_len * num_heads, depth)
  // relu_kernel_transform
  RSSVectorMyType tmp1((B * SL * NH) * NF);
  RSSVectorMyType tmp2((B * SL * NH) * NF);
  auto wqa = *(wq->getActivation());
  funcMatMul(*(wq->getActivation()), projection_matrix, tmp1, (B * SL * NH), D, NF, 0, 1, FLOAT_PRECISION);
  funcDotProduct(tmp1, constRatio, tmp2, (B * SL * NH) * NF, true, FLOAT_PRECISION);
  relu1->forward(tmp2); // (batch_size, seq_len, num_heads, nb_features)
  addVectors(*(relu1->getActivation()), constStab, *(relu1->getActivation()), (B * SL * NH) * NF);

  funcMatMul(*(wk->getActivation()), projection_matrix, tmp1, (B * SL * NH), D, NF, 0, 1, FLOAT_PRECISION);
  funcDotProduct(tmp1, constRatio, tmp2, (B * SL * NH) * NF, true, FLOAT_PRECISION);
  relu2->forward(tmp2); // (batch_size, seq_len, num_heads, nb_features)
  addVectors(*(relu2->getActivation()), constStab, *(relu2->getActivation()), (B * SL * NH) * NF);

  auto reluQ = *(relu1->getActivation());
  auto reluK = *(relu2->getActivation());
  auto V = *(wv->getActivation());

  RSSVectorMyType QKV(B * SL * NH * D);

  if (conf.causal)
  {
    RSSVectorMyType sums(B * NH * NF * D);
    for (size_t l = 0; l < SL; ++l)
    {
      for (size_t b = 0; b < B; ++b)
      {
        for (size_t h = 0; h < NH; ++h)
        {
          RSSVectorMyType K_i(NF);
          copy(reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), K_i.begin());
          RSSVectorMyType V_i(D);
          copy(V.begin() + (b * (SL * NH * D) + l * (NH * D) + h * (D)), V.begin() + (b * (SL * NH * D) + l * (NH * D) + (h + 1) * (D)), V_i.begin());
          RSSVectorMyType KV_i(NF * D);
          funcMatMul(K_i, V_i, KV_i, NF, 1, D, 0, 0, FLOAT_PRECISION);
          for (size_t t = 0; t < NF * D; ++t)
          {
            sums[b * (NH * NF * D) + h * (NF * D) + t] = sums[b * (NH * NF * D) + h * (NF * D) + t] + KV_i[t];
          }

          RSSVectorMyType Q_i(NF);
          copy(reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), Q_i.begin());
          RSSVectorMyType sum(NF * D);
          copy(sums.begin() + (b * (NH * NF * D) + h * (NF * D)), sums.begin() + (b * (NH * NF * D) + (h + 1) * (NF * D)), sum.begin());
          RSSVectorMyType QKV_i(D);
          funcMatMul(Q_i, sum, QKV_i, 1, NF, D, 0, 0, FLOAT_PRECISION);
          copy(QKV_i.begin(), QKV_i.end(), QKV.begin() + b * (SL * NH * D) + l * (NH * D) + h * D);
        }
      }
    }
  }
  else
  {
    // Q(KV)
    // transpose Q (B, L, H, M) -> (B, H, L, M)
    RSSVectorMyType transQ(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            transQ[i * (NH * SL * NF) + k * (SL * NF) + j * NF + l] = reluQ[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose K (B, L, H, M) -> (B, H, M, L)
    RSSVectorMyType transK(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            transK[i * (NH * NF * SL) + k * (NF * SL) + l * SL + j] = reluK[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose V (B, L, H, D) -> (B, H, L, D)
    RSSVectorMyType transV(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            transV[i * (NH * SL * D) + k * (SL * D) + j * D + l] = V[i * (SL * NH * D) + j * (NH * D) + k * D + l];
          }
        }
      }
    }

    // KV (B, H, M, L) * (B, H, L, D)-> (B, H, M, D)
    // RSSVectorMyType KV(B*NH*NF*D);
    RSSVectorMyType inA(NF * SL);
    RSSVectorMyType inB(SL * D);
    RSSVectorMyType out(NF * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(transK.begin() + (i * (NH * NF * SL) + j * (NF * SL)), transK.begin() + (i * (NH * NF * SL) + (j + 1) * (NF * SL)), inA.begin());
        copy(transV.begin() + (i * (NH * SL * D) + j * (SL * D)), transV.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, SL, D, 0, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), KV.begin() + i * (NH * NF * D) + j * (NF * D));
      }
    }

    // QKV (B, H, L, M) * (B, H, M, D) -> (B, H, L, D)
    RSSVectorMyType transQKV(B * NH * SL * D);
    inA.resize(SL * NF);
    inB.resize(NF * D);
    out.resize(SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(transQ.begin() + (i * (NH * SL * NF) + j * (SL * NF)), transQ.begin() + (i * (NH * SL * NF) + (j + 1) * (SL * NF)), inA.begin());
        copy(KV.begin() + (i * (NH * NF * D) + j * (NF * D)), KV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, NF, D, 0, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), transQKV.begin() + i * (NH * SL * D) + j * (SL * D));
      }
    }

    // transpose QKV (B, H, L, D) -> (B, L, H, D)
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        for (size_t k = 0; k < SL; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            QKV[i * (SL * NH * D) + k * (NH * D) + j * D + l] = transQKV[i * (NH * SL * D) + j * (SL * D) + k * D + l];
          }
        }
      }
    }
  }

  // vector<myType> output(QKV.size());
  // funcReconstruct(QKV, output, output.size(), "QKV", true);
  RSSVectorMyType nQKV(QKV.size());
  funcDotProduct(QKV, constAttenNorm, nQKV, nQKV.size(), true, FLOAT_PRECISION);

  // vector<myType> out(nQKV.size());
  // funcReconstruct(nQKV, out, out.size(), "out", true);
  // consider (batch_size, seq_len, num_heads, depth) as (batch_size, seq_len, d_model)
  wo->forward(nQKV);

  auto tmp_act = *(wo->getActivation());
  copy(tmp_act.begin(), tmp_act.end(), activations.begin());
}

// TODO: Recheck backprop after forward bug fixed.
void MHAttention::computeDelta(RSSVectorMyType &prevDelta)
{
  log_print("MHAttention.computeDelta");

  size_t B = conf.batchSize;
  size_t NH = conf.num_heads;
  size_t DM = conf.d_model;
  size_t NF = conf.nb_features;
  size_t SL = conf.seq_len;
  size_t D = conf.depth;
  float attention_normalizer = conf.attention_normalizer;

  RSSVectorMyType dout(B * SL * DM);
  wo->computeDelta(dout);

  RSSVectorMyType dQKV(constAttenNorm.size());
  funcDotProduct(dout, constAttenNorm, dQKV, dQKV.size(), true, FLOAT_PRECISION);

  if (conf.causal)
  {
    cout << "HI" << endl;
  }
  else
  {
    // transpose dQKV (B, L, H, D) -> (B, H, L, D)
    RSSVectorMyType transdQKV(B * NH * SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            transdQKV[i * (NH * SL * D) + k * (SL * D) + j * D + l] = dQKV[i * (SL * NH * D) + j * (NH * D) + k * D + l];
          }
        }
      }
    }

    // dKV (B, H, L, M) * (B, H, L, D) -> (B, H, M, D)
    RSSVectorMyType dKV(B * NH * NF * D);
    auto reluQ = *(relu1->getActivation());
    // transpose Q (B, L, H, M) -> (B, H, L, M)
    RSSVectorMyType transQ(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            transQ[i * (NH * SL * NF) + k * (SL * NF) + j * NF + l] = reluQ[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    RSSVectorMyType inA(SL * NF);
    RSSVectorMyType inB(SL * D);
    RSSVectorMyType out(NF * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(transQ.begin() + (i * (NH * SL * NF) + j * (SL * NF)), transQ.begin() + (i * (NH * SL * NF) + (j + 1) * (SL * NF)), inA.begin());
        copy(transdQKV.begin() + (i * (NH * SL * D) + j * (SL * D)), transdQKV.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, SL, D, 1, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), dKV.begin() + i * (NH * NF * D) + j * (NF * D));
      }
    }

    // dreluQ (B, H, L, D) * (B, H, M, D) -> (B, H, L, M)
    RSSVectorMyType transdreluQ(B * NH * SL * NF);
    inA.resize(SL * D);
    inB.resize(NF * D);
    out.resize(SL * NF);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(transdQKV.begin() + (i * (NH * SL * D) + j * (SL * D)), transdQKV.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inA.begin());
        copy(KV.begin() + (i * (NH * NF * D) + j * (NF * D)), KV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, D, NF, 0, 1, FLOAT_PRECISION);
        copy(out.begin(), out.end(), transdreluQ.begin() + i * (NH * SL * NF) + j * (SL * NF));
      }
    }

    // dreluK (B, H, M, D) * (B, H, L, D) -> (B, H, M, L)
    RSSVectorMyType transdreluK(B * NH * NF * SL);
    // transpose V (B, L, H, D) -> (B, H, L, D)
    auto V = *(wv->getActivation());
    RSSVectorMyType transV(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            transV[i * (NH * SL * D) + k * (SL * D) + j * D + l] = V[i * (SL * NH * D) + j * (NH * D) + k * D + l];
          }
        }
      }
    }
    inA.resize(NF * D);
    inB.resize(SL * D);
    out.resize(NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(dKV.begin() + (i * (NH * NF * D) + j * (NF * D)), dKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inA.begin());
        copy(transV.begin() + (i * (NH * SL * D) + j * (SL * D)), transV.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, D, SL, 0, 1, FLOAT_PRECISION);
        copy(out.begin(), out.end(), transdreluK.begin() + i * (NH * NF * SL) + j * (NF * SL));
      }
    }

    // dV (B, H, M, L) * (B, H, M, D) -> (B, H, L, D)
    RSSVectorMyType dV(B * NH * SL * D);
    auto reluK = *(relu2->getActivation());
    // transpose reluK (B, L, H, M) -> (B, H, M, L)
    RSSVectorMyType transreluK(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            transreluK[i * (NH * NF * SL) + k * (NF * SL) + l * SL + j] = reluK[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }
    inA.resize(NF * SL);
    inB.resize(NF * D);
    out.resize(SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(transreluK.begin() + (i * (NH * NF * SL) + j * (NF * SL)), transreluK.begin() + (i * (NH * NF * SL) + (j + 1) * (NF * SL)), inA.begin());
        copy(dKV.begin() + (i * (NH * NF * D) + j * (NF * D)), dKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, NF, D, 1, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), dV.begin() + i * (NH * SL * D) + j * (SL * D));
      }
    }

    // transpose dreluQ (B, H, L, M) -> (B, L, H, M)
    auto dreluQ = *(relu1->getDelta());
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        for (size_t k = 0; k < SL; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            dreluQ[i * (SL * NH * NF) + k * (NH * NF) + j * NF + l] = transdreluQ[i * (NH * SL * NF) + j * (SL * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose dreluK (B, H, M, L) -> (B, L, H, M)
    auto dreluK = *(relu2->getDelta());
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        for (size_t k = 0; k < NF; ++k)
        {
          for (size_t l = 0; l < SL; ++l)
          {
            dreluK[i * (SL * NH * NF) + l * (NH * NF) + j * NF + k] = transdreluK[i * (NH * NF * SL) + j * (NF * SL) + k * SL + l];
          }
        }
      }
    }

    // ----------------------
    RSSVectorMyType dpQ(B * SL * NH * NF);
    RSSVectorMyType dpK(B * SL * NH * NF);

    relu1->computeDelta(dpQ);
    relu2->computeDelta(dpK);

    vector<smallType> _out(relu1->reluPrime.size());
    funcReconstruct(relu1->reluPrime, _out, _out.size(), "d", true);

    funcDotProduct(*(relu1->getDelta()), constRatio, dpQ, dpQ.size(), true, FLOAT_PRECISION);
    funcDotProduct(*(relu2->getDelta()), constRatio, dpK, dpK.size(), true, FLOAT_PRECISION);

    // dQ (B, L, H, M) * (M, D) -> (B, L, H, D)
    RSSVectorMyType dQ(B * SL * NH * D);
    funcMatMul(dpQ, projection_matrix, dQ, B * SL * NH, NF, D, 0, 0, FLOAT_PRECISION);

    // dK (B, L, H, M) * (M, D) -> (B, L, H, D)
    RSSVectorMyType dK(B * SL * NH * D);
    funcMatMul(dpK, projection_matrix, dK, B * SL * NH, NF, D, 0, 0, FLOAT_PRECISION);

    wq->computeDelta(dQ);
    wk->computeDelta(dK);
    wv->computeDelta(dV);

    addVectors(deltas, *(wq->getDelta()), deltas, deltas.size());
    addVectors(deltas, *(wk->getDelta()), deltas, deltas.size());
    addVectors(deltas, *(wv->getDelta()), deltas, deltas.size());
  }
}

void MHAttention::updateEquations(const RSSVectorMyType &prevActivations)
{
  log_print("MHAttention.updateEquations");
}
