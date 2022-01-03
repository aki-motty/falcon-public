#pragma once
#include "MHAttention.h"
#include "Functionalities.h"
#include "Precompute.h"
#include "secondary.h"

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
      reluKV(conf->batchSize * conf->num_heads * conf->nb_features * conf->depth),
      sums(conf->seq_len * conf->batchSize * conf->num_heads * conf->nb_features * conf->depth),
      nreluQreluKV(conf->batchSize * conf->seq_len * conf->num_heads * conf->depth)
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
  // print_myType(tmp_attenNorm, "const attention norm", "FLOAT");
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

  RSSVectorMyType reluQreluKV(B * SL * NH * D);
  start_m();
  if (conf.causal)
  {
    for (size_t l = 0; l < SL; ++l)
    {
      for (size_t b = 0; b < B; ++b)
      {
        for (size_t h = 0; h < NH; ++h)
        {
          RSSVectorMyType reluK_i(NF);
          copy(reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), reluK_i.begin());
          RSSVectorMyType V_i(D);
          copy(V.begin() + (b * (SL * NH * D) + l * (NH * D) + h * (D)), V.begin() + (b * (SL * NH * D) + l * (NH * D) + (h + 1) * (D)), V_i.begin());
          RSSVectorMyType reluKV_i(NF * D);
          funcMatMul(reluK_i, V_i, reluKV_i, NF, 1, D, 0, 0, FLOAT_PRECISION);
          for (size_t t = 0; t < NF * D; ++t)
          {
            if (l == 0)
              sums[b * (NH * NF * D) + h * (NF * D) + t] = reluKV_i[t];
            else
              sums[l * (B * NH * NF * D) + b * (NH * NF * D) + h * (NF * D) + t] = sums[l * (B * NH * NF * D) + b * (NH * NF * D) + h * (NF * D) + t] + reluKV_i[t];
          }

          RSSVectorMyType reluQ_i(NF);
          copy(reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), reluQ_i.begin());
          copy(sums.begin() + (l * (B * NH * NF * D) + b * (NH * NF * D) + h * (NF * D)), sums.begin() + (l * (B * NH * NF * D) + b * (NH * NF * D) + (h + 1) * (NF * D)), reluKV_i.begin());
          RSSVectorMyType reluQreluKV_i(D);
          funcMatMul(reluQ_i, reluKV_i, reluQreluKV_i, 1, NF, D, 0, 0, FLOAT_PRECISION);
          copy(reluQreluKV_i.begin(), reluQreluKV_i.end(), reluQreluKV.begin() + b * (SL * NH * D) + l * (NH * D) + h * D);
        }
      }
    }
  }
  else
  {
    // Q(KV)
    // transpose Q (B, L, H, M) -> (B, H, L, M)
    RSSVectorMyType t_reluQ(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            t_reluQ[i * (NH * SL * NF) + k * (SL * NF) + j * NF + l] = reluQ[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose K (B, L, H, M) -> (B, H, M, L)
    RSSVectorMyType t_reluK(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            t_reluK[i * (NH * NF * SL) + k * (NF * SL) + l * SL + j] = reluK[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose V (B, L, H, D) -> (B, H, L, D)
    RSSVectorMyType t_V(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            t_V[i * (NH * SL * D) + k * (SL * D) + j * D + l] = V[i * (SL * NH * D) + j * (NH * D) + k * D + l];
          }
        }
      }
    }

    // reluKV (B, H, M, L) * (B, H, L, D)-> (B, H, M, D)
    // RSSVectorMyType reluKV(B*NH*NF*D);
    RSSVectorMyType inA(NF * SL);
    RSSVectorMyType inB(SL * D);
    RSSVectorMyType out(NF * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(t_reluK.begin() + (i * (NH * NF * SL) + j * (NF * SL)), t_reluK.begin() + (i * (NH * NF * SL) + (j + 1) * (NF * SL)), inA.begin());
        copy(t_V.begin() + (i * (NH * SL * D) + j * (SL * D)), t_V.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, SL, D, 0, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), reluKV.begin() + i * (NH * NF * D) + j * (NF * D));
      }
    }

    // QKV (B, H, L, M) * (B, H, M, D) -> (B, H, L, D)
    RSSVectorMyType t_reluQreluKV(B * NH * SL * D);
    inA.resize(SL * NF);
    inB.resize(NF * D);
    out.resize(SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(t_reluQ.begin() + (i * (NH * SL * NF) + j * (SL * NF)), t_reluQ.begin() + (i * (NH * SL * NF) + (j + 1) * (SL * NF)), inA.begin());
        copy(reluKV.begin() + (i * (NH * NF * D) + j * (NF * D)), reluKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, NF, D, 0, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), t_reluQreluKV.begin() + i * (NH * SL * D) + j * (SL * D));
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
            reluQreluKV[i * (SL * NH * D) + k * (NH * D) + j * D + l] = t_reluQreluKV[i * (NH * SL * D) + j * (SL * D) + k * D + l];
          }
        }
      }
    }
  }
  end_m("RELU attention");

  // vector<myType> output(QKV.size());
  // funcReconstruct(QKV, output, output.size(), "QKV", true);
  // RSSVectorMyType nreluQreluKV(reluQreluKV.size());
  funcDotProduct(reluQreluKV, constAttenNorm, nreluQreluKV, nreluQreluKV.size(), true, FLOAT_PRECISION);

  // vector<myType> out(nQKV.size());
  // funcReconstruct(nQKV, out, out.size(), "out", true);
  // consider (batch_size, seq_len, num_heads, depth) as (batch_size, seq_len, d_model)
  wo->forward(nreluQreluKV);

  auto tmp_act = *(wo->getActivation());
  copy(tmp_act.begin(), tmp_act.end(), activations.begin());
}

void MHAttention::forward(const RSSVectorMyType &encoderActivation, const RSSVectorMyType &decoderActivation)
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
  wq->forward(decoderActivation); // (batch_size * seq_len, d_model)
  wk->forward(encoderActivation); // (batch_size * seq_len, d_model)
  wv->forward(encoderActivation); // (batch_size * seq_len, d_model)

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

  RSSVectorMyType reluQreluKV(B * SL * NH * D);

  if (conf.causal)
  {
    for (size_t l = 0; l < SL; ++l)
    {
      for (size_t b = 0; b < B; ++b)
      {
        for (size_t h = 0; h < NH; ++h)
        {
          RSSVectorMyType reluK_i(NF);
          copy(reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), reluK_i.begin());
          RSSVectorMyType V_i(D);
          copy(V.begin() + (b * (SL * NH * D) + l * (NH * D) + h * (D)), V.begin() + (b * (SL * NH * D) + l * (NH * D) + (h + 1) * (D)), V_i.begin());
          RSSVectorMyType reluKV_i(NF * D);
          funcMatMul(reluK_i, V_i, reluKV_i, NF, 1, D, 0, 0, FLOAT_PRECISION);
          for (size_t t = 0; t < NF * D; ++t)
          {
            if (l == 0)
              sums[b * (NH * NF * D) + h * (NF * D) + t] = reluKV_i[t];
            else
              sums[l * (B * NH * NF * D) + b * (NH * NF * D) + h * (NF * D) + t] = sums[l * (B * NH * NF * D) + b * (NH * NF * D) + h * (NF * D) + t] + reluKV_i[t];
          }

          RSSVectorMyType reluQ_i(NF);
          copy(reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), reluQ_i.begin());
          copy(sums.begin() + (l * (B * NH * NF * D) + b * (NH * NF * D) + h * (NF * D)), sums.begin() + (l * (B * NH * NF * D) + b * (NH * NF * D) + (h + 1) * (NF * D)), reluKV_i.begin());
          RSSVectorMyType reluQreluKV_i(D);
          funcMatMul(reluQ_i, reluKV_i, reluQreluKV_i, 1, NF, D, 0, 0, FLOAT_PRECISION);
          copy(reluQreluKV_i.begin(), reluQreluKV_i.end(), reluQreluKV.begin() + b * (SL * NH * D) + l * (NH * D) + h * D);
        }
      }
    }
  }
  else
  {
    // Q(KV)
    // transpose Q (B, L, H, M) -> (B, H, L, M)
    RSSVectorMyType t_reluQ(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            t_reluQ[i * (NH * SL * NF) + k * (SL * NF) + j * NF + l] = reluQ[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose K (B, L, H, M) -> (B, H, M, L)
    RSSVectorMyType t_reluK(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            t_reluK[i * (NH * NF * SL) + k * (NF * SL) + l * SL + j] = reluK[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose V (B, L, H, D) -> (B, H, L, D)
    RSSVectorMyType t_V(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            t_V[i * (NH * SL * D) + k * (SL * D) + j * D + l] = V[i * (SL * NH * D) + j * (NH * D) + k * D + l];
          }
        }
      }
    }

    // reluKV (B, H, M, L) * (B, H, L, D)-> (B, H, M, D)
    // RSSVectorMyType reluKV(B*NH*NF*D);
    RSSVectorMyType inA(NF * SL);
    RSSVectorMyType inB(SL * D);
    RSSVectorMyType out(NF * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(t_reluK.begin() + (i * (NH * NF * SL) + j * (NF * SL)), t_reluK.begin() + (i * (NH * NF * SL) + (j + 1) * (NF * SL)), inA.begin());
        copy(t_V.begin() + (i * (NH * SL * D) + j * (SL * D)), t_V.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, SL, D, 0, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), reluKV.begin() + i * (NH * NF * D) + j * (NF * D));
      }
    }

    // QKV (B, H, L, M) * (B, H, M, D) -> (B, H, L, D)
    RSSVectorMyType t_reluQreluKV(B * NH * SL * D);
    inA.resize(SL * NF);
    inB.resize(NF * D);
    out.resize(SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(t_reluQ.begin() + (i * (NH * SL * NF) + j * (SL * NF)), t_reluQ.begin() + (i * (NH * SL * NF) + (j + 1) * (SL * NF)), inA.begin());
        copy(reluKV.begin() + (i * (NH * NF * D) + j * (NF * D)), reluKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, NF, D, 0, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), t_reluQreluKV.begin() + i * (NH * SL * D) + j * (SL * D));
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
            reluQreluKV[i * (SL * NH * D) + k * (NH * D) + j * D + l] = t_reluQreluKV[i * (NH * SL * D) + j * (SL * D) + k * D + l];
          }
        }
      }
    }
  }

  // vector<myType> output(QKV.size());
  // funcReconstruct(QKV, output, output.size(), "QKV", true);
  // RSSVectorMyType nreluQreluKV(reluQreluKV.size());
  funcDotProduct(reluQreluKV, constAttenNorm, nreluQreluKV, nreluQreluKV.size(), true, FLOAT_PRECISION);

  // vector<myType> out(nQKV.size());
  // funcReconstruct(nQKV, out, out.size(), "out", true);
  // consider (batch_size, seq_len, num_heads, depth) as (batch_size, seq_len, d_model)
  wo->forward(nreluQreluKV);

  auto tmp_act = *(wo->getActivation());
  copy(tmp_act.begin(), tmp_act.end(), activations.begin());
}

// calc prevDelta
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

  RSSVectorMyType d_nreluQreluKV(B * SL * DM);
  wo->computeDelta(d_nreluQreluKV);

  RSSVectorMyType d_reluQreluKV(constAttenNorm.size());
  funcDotProduct(d_nreluQreluKV, constAttenNorm, d_reluQreluKV, d_reluQreluKV.size(), true, FLOAT_PRECISION); // checked

  auto d_V = *(wv->getDelta());
  auto d_reluQ = *(relu1->getDelta());
  auto d_reluK = *(relu2->getDelta());

  auto reluQ = *(relu1->getActivation());
  auto reluK = *(relu2->getActivation());
  auto V = *(wv->getActivation());

  if (conf.causal)
  {
    RSSVectorMyType d_reluKV_sum(B * NH * NF * D);
    for (long l = SL - 1; l > 0; --l)
    {
      for (size_t b = 0; b < B; ++b)
      {
        for (size_t h = 0; h < NH; ++h)
        {
          RSSVectorMyType d_reluQ_i(NF);
          RSSVectorMyType d_reluQreluKV_i(D);
          copy(d_reluQreluKV.begin() + (b * (SL * NH * D) + l * (NH * D) + h * (D)), d_reluQreluKV.begin() + (b * (SL * NH * D) + l * (NH * D) + (h + 1) * (D)), d_reluQreluKV_i.begin());
          RSSVectorMyType sum(NF * D);
          copy(sums.begin() + l * (B * NH * NF * D) + b * (NH * NF * D) + h * (NF * D), sums.begin() + (l * (B * NH * NF * D) + b * (NH * NF * D) + (h + 1) * (NF * D)), sum.begin());
          funcMatMul(d_reluQreluKV_i, sum, d_reluQ_i, 1, D, NF, 0, 1, FLOAT_PRECISION);
          copy(d_reluQ_i.begin(), d_reluQ_i.end(), d_reluQ.begin() + b * (SL * NH * NF) + l * (NH * NF) + h * NF);

          RSSVectorMyType d_reluKV_i(NF * D);
          RSSVectorMyType reluQ_i(NF);
          copy(reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), reluQ_i.begin());
          funcMatMul(reluQ_i, d_reluQreluKV_i, d_reluKV_i, NF, 1, D, 1, 0, FLOAT_PRECISION);

          // add 前のやつ
          for (size_t t = 0; t < NF * D; ++t)
          {
            if (l == SL - 1)
              d_reluKV_sum[b * (NH * NF * D) + h * (NF * D) + t] = d_reluKV_i[t];
            else
              d_reluKV_sum[b * (NH * NF * D) + h * (NF * D) + t] = d_reluKV_sum[b * (NH * NF * D) + h * (NF * D) + t] + d_reluKV_i[t];
          }

          RSSVectorMyType d_reluK_i(NF);
          RSSVectorMyType d_V_i(D);
          RSSVectorMyType d_reluKV_sum_i(NF * D);
          RSSVectorMyType reluK_i(NF);
          RSSVectorMyType V_i(D);
          copy(d_reluKV_sum.begin() + (b * (NH * NF * D) + h * (NF * D)), d_reluKV_sum.begin() + (b * (NH * NF * D) + (h + 1) * (NF * D)), d_reluKV_sum_i.begin());
          copy(reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), reluK_i.begin());
          copy(V.begin() + (b * (SL * NH * D) + l * (NH * D) + h * (D)), V.begin() + (b * (SL * NH * D) + l * (NH * D) + (h + 1) * (D)), V_i.begin());
          funcMatMul(d_reluKV_sum_i, V_i, d_reluK_i, NF, D, 1, 0, 1, FLOAT_PRECISION);
          funcMatMul(reluK_i, d_reluKV_sum_i, d_V_i, 1, NF, D, 1, 0, FLOAT_PRECISION);
          copy(d_reluK_i.begin(), d_reluK_i.end(), d_reluK.begin() + b * (SL * NH * NF) + l * (NH * NF) + h * NF);
          copy(d_V_i.begin(), d_V_i.end(), d_V.begin() + b * (SL * NH * D) + l * (NH * D) + h * D);
        }
      }
    }
  }
  else
  {
    // transpose d_reluQreluKV (B, L, H, D) -> (B, H, L, D)
    RSSVectorMyType t_d_reluQreluKV(B * NH * SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            t_d_reluQreluKV[i * (NH * SL * D) + k * (SL * D) + j * D + l] = d_reluQreluKV[i * (SL * NH * D) + j * (NH * D) + k * D + l];
          }
        }
      }
    }

    // t_reluQ * t_d_reluQreluKV = d_reluKV : (B, H, M, L) * (B, H, L, D) -> (B, H, M, D)
    RSSVectorMyType d_reluKV(B * NH * NF * D);

    // transpose reluQ (B, L, H, M) -> (B, H, M, L)
    RSSVectorMyType t_reluQ(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            t_reluQ[i * (NH * NF * SL) + k * (NF * SL) + l * SL + j] = reluQ[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    RSSVectorMyType inA(NF * SL);
    RSSVectorMyType inB(SL * D);
    RSSVectorMyType out(NF * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(t_reluQ.begin() + (i * (NH * NF * SL) + j * (NF * SL)), t_reluQ.begin() + (i * (NH * NF * SL) + (j + 1) * (NF * SL)), inA.begin());
        copy(t_d_reluQreluKV.begin() + (i * (NH * SL * D) + j * (SL * D)), t_d_reluQreluKV.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, SL, D, 0, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), d_reluKV.begin() + i * (NH * NF * D) + j * (NF * D));
      }
    }

    // t_d_reluQreluKV * reluKV = d_reluQ (B, H, L, D) * (B, H, M, D) -> (B, H, L, M)
    RSSVectorMyType t_d_reluQ(B * NH * SL * NF);
    inA.resize(SL * D);
    inB.resize(NF * D);
    out.resize(SL * NF);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(t_d_reluQreluKV.begin() + (i * (NH * SL * D) + j * (SL * D)), t_d_reluQreluKV.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inA.begin());
        copy(reluKV.begin() + (i * (NH * NF * D) + j * (NF * D)), reluKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, D, NF, 0, 1, FLOAT_PRECISION);
        copy(out.begin(), out.end(), t_d_reluQ.begin() + i * (NH * SL * NF) + j * (SL * NF));
      }
    }

    // d_reluKV * t_V = t_d_reluK (B, H, M, D) * (B, H, L, D) -> (B, H, M, L)
    RSSVectorMyType t_d_reluK(B * NH * NF * SL);
    // transpose V (B, L, H, D) -> (B, H, L, D)

    RSSVectorMyType t_V(B * NH * SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            t_V[i * (NH * SL * D) + k * (SL * D) + j * D + l] = V[i * (SL * NH * D) + j * (NH * D) + k * D + l];
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
        copy(d_reluKV.begin() + (i * (NH * NF * D) + j * (NF * D)), d_reluKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inA.begin());
        copy(t_V.begin() + (i * (NH * SL * D) + j * (SL * D)), t_V.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, D, SL, 0, 1, FLOAT_PRECISION);
        copy(out.begin(), out.end(), t_d_reluK.begin() + i * (NH * NF * SL) + j * (NF * SL));
      }
    }

    // t_reluK * d_reluKV = d_V : (B, H, M, L) * (B, H, M, D) -> (B, H, L, D)
    // RSSVectorMyType d_V(B * NH * SL * D);

    // transpose reluK (B, L, H, M) -> (B, H, M, L)
    RSSVectorMyType t_reluK(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            t_reluK[i * (NH * NF * SL) + k * (NF * SL) + l * SL + j] = reluK[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
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
        copy(t_reluK.begin() + (i * (NH * NF * SL) + j * (NF * SL)), t_reluK.begin() + (i * (NH * NF * SL) + (j + 1) * (NF * SL)), inA.begin());
        copy(d_reluKV.begin() + (i * (NH * NF * D) + j * (NF * D)), d_reluKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, NF, D, 1, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), d_V.begin() + i * (NH * SL * D) + j * (SL * D));
      }
    }

    // transpose t_d_reluQ (B, H, L, M) -> (B, L, H, M)

    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        for (size_t k = 0; k < SL; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            d_reluQ[i * (SL * NH * NF) + k * (NH * NF) + j * NF + l] = t_d_reluQ[i * (NH * SL * NF) + j * (SL * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose d_reluK (B, H, M, L) -> (B, L, H, M)

    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        for (size_t k = 0; k < NF; ++k)
        {
          for (size_t l = 0; l < SL; ++l)
          {
            d_reluK[i * (SL * NH * NF) + l * (NH * NF) + j * NF + k] = t_d_reluK[i * (NH * NF * SL) + j * (NF * SL) + k * SL + l];
          }
        }
      }
    }
  }
  // ----------------------
  RSSVectorMyType d_npQ(B * SL * NH * NF);
  RSSVectorMyType d_npK(B * SL * NH * NF);

  relu1->computeDelta(d_npQ);
  relu2->computeDelta(d_npK);

  RSSVectorMyType d_pQ(B * SL * NH * NF);
  RSSVectorMyType d_pK(B * SL * NH * NF);

  // vector<smallType> _out(relu1->reluPrime.size());
  // funcReconstruct(relu1->reluPrime, _out, _out.size(), "d", true);

  funcDotProduct(d_npQ, constRatio, d_pQ, d_pQ.size(), true, FLOAT_PRECISION);
  funcDotProduct(d_npK, constRatio, d_pK, d_pK.size(), true, FLOAT_PRECISION);

  // dQ (B, L, H, M) * (M, D) -> (B, L, H, D)
  funcMatMul(d_pQ, projection_matrix, *(wq->getDelta()), B * SL * NH, NF, D, 0, 0, FLOAT_PRECISION);

  // dK (B, L, H, M) * (M, D) -> (B, L, H, D)
  funcMatMul(d_pK, projection_matrix, *(wk->getDelta()), B * SL * NH, NF, D, 0, 0, FLOAT_PRECISION);

  RSSVectorMyType tmp(B * SL * DM);
  wq->computeDelta(tmp);
  addVectors(prevDelta, tmp, prevDelta, tmp.size());
  wk->computeDelta(tmp);
  addVectors(prevDelta, tmp, prevDelta, tmp.size());
  wv->computeDelta(tmp);
  addVectors(prevDelta, tmp, prevDelta, tmp.size());
}

void MHAttention::computeDelta(RSSVectorMyType &prevEncoderDelta, RSSVectorMyType &prevDecoderDelta)
{
  log_print("MHAttention.computeDelta");

  size_t B = conf.batchSize;
  size_t NH = conf.num_heads;
  size_t DM = conf.d_model;
  size_t NF = conf.nb_features;
  size_t SL = conf.seq_len;
  size_t D = conf.depth;
  float attention_normalizer = conf.attention_normalizer;

  RSSVectorMyType d_nreluQreluKV(B * SL * DM);
  wo->computeDelta(d_nreluQreluKV);

  RSSVectorMyType d_reluQreluKV(constAttenNorm.size());
  funcDotProduct(d_nreluQreluKV, constAttenNorm, d_reluQreluKV, d_reluQreluKV.size(), true, FLOAT_PRECISION); // checked

  auto d_V = *(wv->getDelta());
  auto d_reluQ = *(relu1->getDelta());
  auto d_reluK = *(relu2->getDelta());

  auto reluQ = *(relu1->getActivation());
  auto reluK = *(relu2->getActivation());
  auto V = *(wv->getActivation());

  if (conf.causal)
  {
    RSSVectorMyType d_reluKV_sum(B * NH * NF * D);
    for (long l = SL - 1; l > 0; --l)
    {
      for (size_t b = 0; b < B; ++b)
      {
        for (size_t h = 0; h < NH; ++h)
        {
          RSSVectorMyType d_reluQ_i(NF);
          RSSVectorMyType d_reluQreluKV_i(D);
          copy(d_reluQreluKV.begin() + (b * (SL * NH * D) + l * (NH * D) + h * (D)), d_reluQreluKV.begin() + (b * (SL * NH * D) + l * (NH * D) + (h + 1) * (D)), d_reluQreluKV_i.begin());
          RSSVectorMyType sum(NF * D);
          copy(sums.begin() + l * (B * NH * NF * D) + b * (NH * NF * D) + h * (NF * D), sums.begin() + (l * (B * NH * NF * D) + b * (NH * NF * D) + (h + 1) * (NF * D)), sum.begin());
          funcMatMul(d_reluQreluKV_i, sum, d_reluQ_i, 1, D, NF, 0, 1, FLOAT_PRECISION);
          copy(d_reluQ_i.begin(), d_reluQ_i.end(), d_reluQ.begin() + b * (SL * NH * NF) + l * (NH * NF) + h * NF);

          RSSVectorMyType d_reluKV_i(NF * D);
          RSSVectorMyType reluQ_i(NF);
          copy(reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluQ.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), reluQ_i.begin());
          funcMatMul(reluQ_i, d_reluQreluKV_i, d_reluKV_i, NF, 1, D, 1, 0, FLOAT_PRECISION);

          // add 前のやつ
          for (size_t t = 0; t < NF * D; ++t)
          {
            if (l == SL - 1)
              d_reluKV_sum[b * (NH * NF * D) + h * (NF * D) + t] = d_reluKV_i[t];
            else
              d_reluKV_sum[b * (NH * NF * D) + h * (NF * D) + t] = d_reluKV_sum[b * (NH * NF * D) + h * (NF * D) + t] + d_reluKV_i[t];
          }

          RSSVectorMyType d_reluK_i(NF);
          RSSVectorMyType d_V_i(D);
          RSSVectorMyType d_reluKV_sum_i(NF * D);
          RSSVectorMyType reluK_i(NF);
          RSSVectorMyType V_i(D);
          copy(d_reluKV_sum.begin() + (b * (NH * NF * D) + h * (NF * D)), d_reluKV_sum.begin() + (b * (NH * NF * D) + (h + 1) * (NF * D)), d_reluKV_sum_i.begin());
          copy(reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + h * (NF)), reluK.begin() + (b * (SL * NH * NF) + l * (NH * NF) + (h + 1) * (NF)), reluK_i.begin());
          copy(V.begin() + (b * (SL * NH * D) + l * (NH * D) + h * (D)), V.begin() + (b * (SL * NH * D) + l * (NH * D) + (h + 1) * (D)), V_i.begin());
          funcMatMul(d_reluKV_sum_i, V_i, d_reluK_i, NF, D, 1, 0, 1, FLOAT_PRECISION);
          funcMatMul(reluK_i, d_reluKV_sum_i, d_V_i, 1, NF, D, 1, 0, FLOAT_PRECISION);
          copy(d_reluK_i.begin(), d_reluK_i.end(), d_reluK.begin() + b * (SL * NH * NF) + l * (NH * NF) + h * NF);
          copy(d_V_i.begin(), d_V_i.end(), d_V.begin() + b * (SL * NH * D) + l * (NH * D) + h * D);
        }
      }
    }
  }
  else
  {
    // transpose d_reluQreluKV (B, L, H, D) -> (B, H, L, D)
    RSSVectorMyType t_d_reluQreluKV(B * NH * SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            t_d_reluQreluKV[i * (NH * SL * D) + k * (SL * D) + j * D + l] = d_reluQreluKV[i * (SL * NH * D) + j * (NH * D) + k * D + l];
          }
        }
      }
    }

    // t_reluQ * t_d_reluQreluKV = d_reluKV : (B, H, M, L) * (B, H, L, D) -> (B, H, M, D)
    RSSVectorMyType d_reluKV(B * NH * NF * D);

    // transpose reluQ (B, L, H, M) -> (B, H, M, L)
    RSSVectorMyType t_reluQ(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            t_reluQ[i * (NH * NF * SL) + k * (NF * SL) + l * SL + j] = reluQ[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
          }
        }
      }
    }

    RSSVectorMyType inA(NF * SL);
    RSSVectorMyType inB(SL * D);
    RSSVectorMyType out(NF * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(t_reluQ.begin() + (i * (NH * NF * SL) + j * (NF * SL)), t_reluQ.begin() + (i * (NH * NF * SL) + (j + 1) * (NF * SL)), inA.begin());
        copy(t_d_reluQreluKV.begin() + (i * (NH * SL * D) + j * (SL * D)), t_d_reluQreluKV.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, SL, D, 0, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), d_reluKV.begin() + i * (NH * NF * D) + j * (NF * D));
      }
    }

    // t_d_reluQreluKV * reluKV = d_reluQ (B, H, L, D) * (B, H, M, D) -> (B, H, L, M)
    RSSVectorMyType t_d_reluQ(B * NH * SL * NF);
    inA.resize(SL * D);
    inB.resize(NF * D);
    out.resize(SL * NF);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        copy(t_d_reluQreluKV.begin() + (i * (NH * SL * D) + j * (SL * D)), t_d_reluQreluKV.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inA.begin());
        copy(reluKV.begin() + (i * (NH * NF * D) + j * (NF * D)), reluKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, D, NF, 0, 1, FLOAT_PRECISION);
        copy(out.begin(), out.end(), t_d_reluQ.begin() + i * (NH * SL * NF) + j * (SL * NF));
      }
    }

    // d_reluKV * t_V = t_d_reluK (B, H, M, D) * (B, H, L, D) -> (B, H, M, L)
    RSSVectorMyType t_d_reluK(B * NH * NF * SL);
    // transpose V (B, L, H, D) -> (B, H, L, D)

    RSSVectorMyType t_V(B * NH * SL * D);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < D; ++l)
          {
            t_V[i * (NH * SL * D) + k * (SL * D) + j * D + l] = V[i * (SL * NH * D) + j * (NH * D) + k * D + l];
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
        copy(d_reluKV.begin() + (i * (NH * NF * D) + j * (NF * D)), d_reluKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inA.begin());
        copy(t_V.begin() + (i * (NH * SL * D) + j * (SL * D)), t_V.begin() + (i * (NH * SL * D) + (j + 1) * (SL * D)), inB.begin());
        funcMatMul(inA, inB, out, NF, D, SL, 0, 1, FLOAT_PRECISION);
        copy(out.begin(), out.end(), t_d_reluK.begin() + i * (NH * NF * SL) + j * (NF * SL));
      }
    }

    // t_reluK * d_reluKV = d_V : (B, H, M, L) * (B, H, M, D) -> (B, H, L, D)
    // RSSVectorMyType d_V(B * NH * SL * D);

    // transpose reluK (B, L, H, M) -> (B, H, M, L)
    RSSVectorMyType t_reluK(B * NH * NF * SL);
    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < SL; ++j)
      {
        for (size_t k = 0; k < NH; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            t_reluK[i * (NH * NF * SL) + k * (NF * SL) + l * SL + j] = reluK[i * (SL * NH * NF) + j * (NH * NF) + k * NF + l];
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
        copy(t_reluK.begin() + (i * (NH * NF * SL) + j * (NF * SL)), t_reluK.begin() + (i * (NH * NF * SL) + (j + 1) * (NF * SL)), inA.begin());
        copy(d_reluKV.begin() + (i * (NH * NF * D) + j * (NF * D)), d_reluKV.begin() + (i * (NH * NF * D) + (j + 1) * (NF * D)), inB.begin());
        funcMatMul(inA, inB, out, SL, NF, D, 1, 0, FLOAT_PRECISION);
        copy(out.begin(), out.end(), d_V.begin() + i * (NH * SL * D) + j * (SL * D));
      }
    }

    // transpose t_d_reluQ (B, H, L, M) -> (B, L, H, M)

    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        for (size_t k = 0; k < SL; ++k)
        {
          for (size_t l = 0; l < NF; ++l)
          {
            d_reluQ[i * (SL * NH * NF) + k * (NH * NF) + j * NF + l] = t_d_reluQ[i * (NH * SL * NF) + j * (SL * NF) + k * NF + l];
          }
        }
      }
    }

    // transpose d_reluK (B, H, M, L) -> (B, L, H, M)

    for (size_t i = 0; i < B; ++i)
    {
      for (size_t j = 0; j < NH; ++j)
      {
        for (size_t k = 0; k < NF; ++k)
        {
          for (size_t l = 0; l < SL; ++l)
          {
            d_reluK[i * (SL * NH * NF) + l * (NH * NF) + j * NF + k] = t_d_reluK[i * (NH * NF * SL) + j * (NF * SL) + k * SL + l];
          }
        }
      }
    }
  }
  // ----------------------
  RSSVectorMyType d_npQ(B * SL * NH * NF);
  RSSVectorMyType d_npK(B * SL * NH * NF);

  relu1->computeDelta(d_npQ);
  relu2->computeDelta(d_npK);

  RSSVectorMyType d_pQ(B * SL * NH * NF);
  RSSVectorMyType d_pK(B * SL * NH * NF);

  // vector<smallType> _out(relu1->reluPrime.size());
  // funcReconstruct(relu1->reluPrime, _out, _out.size(), "d", true);

  funcDotProduct(d_npQ, constRatio, d_pQ, d_pQ.size(), true, FLOAT_PRECISION);
  funcDotProduct(d_npK, constRatio, d_pK, d_pK.size(), true, FLOAT_PRECISION);

  // dQ (B, L, H, M) * (M, D) -> (B, L, H, D)
  funcMatMul(d_pQ, projection_matrix, *(wq->getDelta()), B * SL * NH, NF, D, 0, 0, FLOAT_PRECISION);

  // dK (B, L, H, M) * (M, D) -> (B, L, H, D)
  funcMatMul(d_pK, projection_matrix, *(wk->getDelta()), B * SL * NH, NF, D, 0, 0, FLOAT_PRECISION);

  RSSVectorMyType tmp(B * SL * DM);
  wq->computeDelta(tmp);
  addVectors(prevDecoderDelta, tmp, prevDecoderDelta, tmp.size());
  wk->computeDelta(tmp);
  addVectors(prevEncoderDelta, tmp, prevEncoderDelta, tmp.size());
  wv->computeDelta(tmp);
  addVectors(prevEncoderDelta, tmp, prevEncoderDelta, tmp.size());
}

void MHAttention::updateEquations(const RSSVectorMyType &prevActivations)
{
  log_print("MHAttention.updateEquations");

  wq->updateEquations(prevActivations);
  wk->updateEquations(prevActivations);
  wv->updateEquations(prevActivations);
  wo->updateEquations(nreluQreluKV);
}

void MHAttention::updateEquations(const RSSVectorMyType &prevEncoderActivations, const RSSVectorMyType &prevDecoderActivations)
{
  log_print("MHAttention.updateEquations");

  wq->updateEquations(prevDecoderActivations);
  wk->updateEquations(prevEncoderActivations);
  wv->updateEquations(prevEncoderActivations);
  wo->updateEquations(nreluQreluKV);
}
