# -*- coding: utf-8 -*-
"""
增强版帕金森病语音特征分析系统 + 多模态交叉注意力融合 + 不确定性加权多任务学习
Enhanced Parkinson's Disease Voice Analysis with Multi-Modal Cross-Attention Fusion
and Uncertainty-Aware Multi-Task Learning

创新点:
1. 多模态交叉注意力融合机制 (Multi-Modal Cross-Attention Fusion, MM-CAF)
2. 不确定性加权的多任务损失 (Uncertainty-Aware Multi-Task Loss Strategy)
"""

# --- 核心库导入 ---
import os
import traceback
from pathlib import Path
from typing import List

from tqdm import tqdm
# --- 数据处理与机器学习 ---
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score)

# --- 深度学习 (TensorFlow/Keras) ---
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# --- 音频处理 ---
import librosa
import librosa.display

import pickle
from datetime import datetime
from pathlib import Path

# --- 可视化 ---
from P4_vis import SCIPaperVisualization

# --- 可选库导入与可用性检查 ---
try:
    from tensorflow_addons.optimizers import AdamW

    print("Using AdamW from tensorflow_addons.")
except ImportError:
    print("Warning: tensorflow_addons not found. Falling back to tf.keras.optimizers.AdamW.")
    from tensorflow.keras.optimizers import AdamW

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not found. SMOTE will be disabled.")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not found. Model interpretability will be limited.")

try:
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Processor, Wav2Vec2Model,trainer

    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch available. Using device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/Transformers not found. Wav2Vec features will be disabled.")

# --- 配置参数 ---
CONFIG = {
    'DATA_PATH': r'D:\data',  # ← 只到根目录
    'OUTPUT_PATH': r'./results',
    'TARGET_COLUMN': 'status',
    'SAMPLE_RATE': 16000,
    'RANDOM_STATE': 42,
    'BATCH_SIZE': 16,  # 家用 GPU 可再调小
    'EPOCHS': 60,  # 小数据不用 100
    'LEARNING_RATE': 3e-4,
    'WEIGHT_DECAY': 1e-4,
    'PATIENCE': 10,
    'SCALE_DATA': True,
    'USE_SMOTE': True,
    'SMOTE_K_NEIGHBORS': 3,  # 样本少时 k 要小于最小类
    'N_SPLITS': 2,  # 2→5，更稳健
    'MAX_FILES_PER_CLASS': None,  # 想快速调试可设 50
    'USE_CONTRASTIVE_LOSS': True,
    'CONTRASTIVE_LOSS_WEIGHT': 0.4,
    'CONTRASTIVE_TEMPERATURE': 0.1,
    'INITIAL_TEMP_SCALE': 10.0,
    'TRAINABLE_TEMP': True,
    'RUN_ABLATION_STUDY': False,  # 先关，跑通再开
    'ABLATION_CONFIGS': [
        {'EXP_NAME': 'MM_CAF_Uncertainty', 'ENCODER_TYPE': 'mm_caf'},
    ],
    'ENCODER_TYPE': 'mm_caf',
    'EMBEDDING_DIM': 256,  # 768→256，小数据降维
    'USE_CLINICAL_CENTER_LOSS': False,  # 没有 UPDRS 文件先关
    'CLINICAL_LOSS_WEIGHT': 0.0,
    'USE_UNCERTAINTY_WEIGHTING': True,
    'HANDCRAFTED_DIM': 40,
    'DEEP_FEATURE_DIM': 768,
    # 新增的优化配置
    'FAST_MODE': True,  # 启用快速模式
    'AUDIO_SEGMENT_LENGTH': 3,  # 音频段长度（秒）
    'MAX_FILES_PER_CLASS': None,  # 每类最大文件数（None表示不限制）
    'SKIP_WAV2VEC': False,  # 是否跳过Wav2Vec特征提取

'USE_CACHE': True,  # 是否启用缓存功能
    'CACHE_DIR': 'cache',  # 缓存目录名
    'AUTO_USE_CACHE': False,  # 是否自动使用缓存（不询问用户）
}



# --- 全局设置 ---
tf.random.set_seed(CONFIG['RANDOM_STATE'])
np.random.seed(CONFIG['RANDOM_STATE'])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# -------------- 新增批量 GPU 接口 --------------
def extract_batch(self, wav_list: List[np.ndarray]) -> np.ndarray:
    """一次把 32×30 s 语音喂给 GPU，返回 32×768 向量"""
    # 统一长度（30 s）避免动态 padding 拖慢 CUDA kernel
    wav_list = [librosa.util.fix_length(w, 16000*30) for w in wav_list]
    inputs = self.processor(
        wav_list, sampling_rate=16_000,
        return_tensors="pt", padding=False).to(DEVICE)   # 整批放 GPU
    with torch.no_grad():
        hidden = self.backbone(**inputs).last_hidden_state        # B×T×768
    emb = self.pool(hidden.transpose(1, 2)).squeeze(-1)          # B×768
    return emb.cpu().numpy()

# 放在 load_data_from_wav_files_optimized 之后
def load_uci_updrs(csv_path):
    """
    仅返回 43-D 特征 + motor_UPDRS
    特征顺序与 extract_audio_features 保持一致即可
    """
    df = pd.read_csv(csv_path)
    # 手工特征列（与 wav 侧 43-D 对应）
    feat_cols = ['Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP',
                 'Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA',
                 'NHR','HNR','RPDE','DFA','PPE',
                 'f0_mean','f0_std','f0_range',          # 下面 3 列需要你自己映射
                 'spectral_centroid_mean','spectral_centroid_std',
                 'spectral_bandwidth_mean','spectral_bandwidth_std',
                 'spectral_rolloff_mean','spectral_rolloff_std',
                 'zcr_mean','zcr_std',
                 'rms_mean','rms_std',
                 'shimmer','jitter',
                 'kurtosis','skewness',
                 'mfcc_1_mean','mfcc_1_std','mfcc_2_mean','mfcc_2_std',
                 'mfcc_3_mean','mfcc_3_std','mfcc_4_mean','mfcc_4_std',
                 'mfcc_5_mean','mfcc_5_std','mfcc_6_mean','mfcc_6_std',
                 'mfcc_7_mean','mfcc_7_std','mfcc_8_mean','mfcc_8_std']
    # 缺啥补 0；这里只列出 UCI 有的，共 22 列，其余补 0 → 43-D
    uci_avail = ['Jitter(%)','Jitter:A','Jitter:RAP','Jitter:PPQ5','Jitter:DDP',
                 'Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA',
                 'NHR','HNR','RPDE','DFA','PPE']
    X = df[uci_avail].values
    # 补到 43-D
    pad = np.zeros((X.shape[0], 43 - X.shape[1]))
    X = np.hstack([X, pad])
    y_updrs = df['motor_UPDRS'].values.astype(np.float32)
    return X, y_updrs


class SeverityTrainer:
    def __init__(self, config, encoder_model):
        self.config = config
        self.encoder = encoder_model          # 冻结的 encoder（来自 wav 训练）
        self.scaler = StandardScaler()
        self.results = {}

    def build_severity_head(self):
        return tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(1)   # 预测 motor_UPDRS
        ], name='severity_head')

    def train_eval(self, X_uci, y_uci, n_fold=5):
        kf = KFold(n_splits=n_fold, shuffle=True, random_state=self.config['RANDOM_STATE'])
        all_true, all_pred = [], []
        for fold, (tr, va) in enumerate(kf.split(X_uci)):
            # 标准化
            X_tr = self.scaler.fit_transform(X_uci[tr])
            X_va = self.scaler.transform(X_uci[va])
            y_tr, y_va = y_uci[tr], y_uci[va]

            # 冻结 encoder + 新建头
            head = self.build_severity_head()
            inputs = layers.Input(shape=(self.config['EMBEDDING_DIM'],))
            out = head(inputs)
            model = Model(inputs, out)
            model.compile(optimizer=AdamW(learning_rate=1e-4), loss='mse', metrics=['mae'])

            # 生成嵌入（只跑一遍，省 GPU）
            emb_tr = self.encoder({'handcrafted': tf.constant(X_tr, dtype=tf.float32),
                                   'raw_audio':  tf.zeros((len(X_tr), 480000))}, training=False)
            emb_va = self.encoder({'handcrafted': tf.constant(X_va, dtype=tf.float32),
                                   'raw_audio':  tf.zeros((len(X_va), 480000))}, training=False)

            # 训练 severity 头
            cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            model.fit(emb_tr, y_tr, validation_data=(emb_va, y_va),
                      epochs=60, batch_size=32, callbacks=[cb], verbose=0)

            pred = model.predict(emb_va, verbose=0).flatten()
            all_true.extend(y_va)
            all_pred.extend(pred)

        self.results = {'true': np.array(all_true),
                        'pred': np.array(all_pred),
                        'r':  np.corrcoef(all_true, all_pred)[0,1],
                        'mae': np.mean(np.abs(np.array(all_true) - np.array(all_pred)))}
        return self.results

# --- 创新点1: 多模态交叉注意力融合机制 ---
# --- 修复后的多模态交叉注意力融合机制 ---
class MultiModalCrossAttentionFusion(layers.Layer):
    """
    多模态交叉注意力融合层
    使用深度特征作为Query，传统声学特征作为Key和Value
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout

        # 特征投影层
        self.deep_projection = layers.Dense(d_model, name='deep_projection')
        self.handcrafted_projection = layers.Dense(d_model, name='handcrafted_projection')

        # 多头交叉注意力 - 修复维度问题
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout,
            name='cross_attention'
        )

        # 前馈网络
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ], name='ffn')

        # 层归一化
        self.ln1 = layers.LayerNormalization(name='ln1')
        self.ln2 = layers.LayerNormalization(name='ln2')
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        """构建层"""
        super().build(input_shape)

    def call(self, inputs, training=False, return_attention_scores=False):
        """
        inputs: [deep_features, handcrafted_features]
        """
        deep_features, handcrafted_features = inputs

        # 确保输入维度正确
        if len(tf.shape(deep_features)) == 1:
            deep_features = tf.expand_dims(deep_features, 0)
        if len(tf.shape(handcrafted_features)) == 1:
            handcrafted_features = tf.expand_dims(handcrafted_features, 0)

        # 投影到相同维度
        deep_proj = self.deep_projection(deep_features)  # (B, d_model)
        handcrafted_proj = self.handcrafted_projection(handcrafted_features)  # (B, d_model)

        # 为交叉注意力添加序列维度
        query = tf.expand_dims(deep_proj, axis=1)  # (B, 1, d_model)
        key = tf.expand_dims(handcrafted_proj, axis=1)  # (B, 1, d_model)
        value = tf.expand_dims(handcrafted_proj, axis=1)  # (B, 1, d_model)

        # 交叉注意力：深度特征查询传统特征
        attention_output = self.cross_attention(
            query=query,
            key=key,
            value=value,
            training=training,
            return_attention_scores=return_attention_scores
        )

        # 处理返回值
        if return_attention_scores:
            attention_output, attention_scores = attention_output
            # 移除序列维度
            attention_output = tf.squeeze(attention_output, axis=1)  # (B, d_model)
            # 处理注意力分数维度
            attention_scores = tf.reduce_mean(attention_scores, axis=[1, 2])  # (B, num_heads)
        else:
            # 移除序列维度
            attention_output = tf.squeeze(attention_output, axis=1)  # (B, d_model)

        # 残差连接和层归一化
        x = self.ln1(deep_proj + self.dropout(attention_output, training=training))

        # 前馈网络
        ffn_output = self.ffn(x, training=training)
        output = self.ln2(x + self.dropout(ffn_output, training=training))

        if return_attention_scores:
            return output, attention_scores
        return output


# --- 创新点2: 不确定性加权多任务损失 ---
# --- 修复后的不确定性加权多任务损失 ---
class UncertaintyWeightedLoss(layers.Layer):
    """
    不确定性加权多任务损失
    基于 Kendall et al. "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
    """

    def __init__(self, num_tasks=3, **kwargs):
        super().__init__(**kwargs)
        self.num_tasks = num_tasks

        # 学习每个任务的不确定性参数 (log variance)
        self.log_vars = self.add_weight(
            name='log_vars',
            shape=(num_tasks,),
            initializer='zeros',
            trainable=True
        )

    def build(self, input_shape):
        """构建层"""
        super().build(input_shape)

    def call(self, losses):
        """
        losses: [classification_loss, contrastive_loss, clinical_loss]
        """
        weighted_losses = []
        precision_losses = []

        for i, loss in enumerate(losses):
            # 确保loss是张量
            loss_tensor = tf.convert_to_tensor(loss, dtype=tf.float32)

            # 计算精度 (1/variance)
            precision = tf.exp(-self.log_vars[i])

            # 加权损失
            weighted_loss = precision * loss_tensor
            weighted_losses.append(weighted_loss)

            # 正则化项 (log variance)
            precision_losses.append(self.log_vars[i])

        # 总损失 = 加权损失之和 + 正则化项
        total_weighted_loss = tf.add_n(weighted_losses)
        regularization = tf.add_n(precision_losses)

        return total_weighted_loss + regularization, weighted_losses, precision_losses


# --- Wav2Vec特征提取器 ---
if TORCH_AVAILABLE:
    class Wav2VecEncoder:
        def __init__(self, model_name="E:/wav2vec2-base-960h"):
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.backbone = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE)
            self.backbone.eval()
            self.pool = nn.AdaptiveAvgPool1d(1)

        def extract_features(self, wav_16k):
            """提取Wav2Vec特征"""
            inputs = self.processor(wav_16k, sampling_rate=16000, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                hidden = self.backbone(**inputs).last_hidden_state
            hidden = hidden.transpose(1, 2)
            emb = self.pool(hidden).squeeze(-1)
            return emb.cpu().numpy()


# --- 多模态编码器 ---
# --- 修复后的多模态编码器 ---
class MultiModalEncoder(Model):
    """
    多模态编码器，结合传统声学特征和深度特征
    """

    def __init__(self, handcrafted_dim=40, deep_dim=768, embedding_dim=512,
                 num_heads=8, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.handcrafted_dim = handcrafted_dim
        self.deep_dim = deep_dim
        self.embedding_dim = embedding_dim

        # Wav2Vec特征提取器
        if TORCH_AVAILABLE:
            self.wav2vec = Wav2VecEncoder()

        # 多模态交叉注意力融合
        self.mm_caf = MultiModalCrossAttentionFusion(
            d_model=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 最终投影层
        self.final_projection = tf.keras.Sequential([
            layers.Dense(embedding_dim, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(embedding_dim)
        ])

    def build(self, input_shape):
        """构建模型"""
        super().build(input_shape)

    def call(self, inputs, training=False, return_attention_scores=False):
        """
        inputs: {'handcrafted': (B, handcrafted_dim), 'raw_audio': (B, T)}
        """
        handcrafted_features = inputs['handcrafted']

        # 提取深度特征
        if TORCH_AVAILABLE and 'raw_audio' in inputs:
            try:
                raw_audio = inputs['raw_audio'].numpy()
                deep_features_list = []

                # 批量处理音频特征提取
                batch_size = len(raw_audio)
                for i in range(batch_size):
                    try:
                        audio = raw_audio[i]
                        # 确保音频长度合适 - 减少处理时间
                        if len(audio) > 48000:  # 3秒@16kHz
                            audio = audio[:48000]
                        elif len(audio) < 16000:  # 最小1秒
                            audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')

                        feat = self.wav2vec.extract_features(audio)
                        if len(feat.shape) == 1:
                            feat = feat.reshape(1, -1)
                        deep_features_list.append(feat[0])
                    except Exception as e:
                        print(f"Warning: Audio {i} processing failed: {e}")
                        # 使用零向量作为后备
                        deep_features_list.append(np.zeros(self.deep_dim))

                deep_features = tf.constant(np.array(deep_features_list), dtype=tf.float32)
            except Exception as e:
                print(f"Warning: Wav2Vec feature extraction failed: {e}")
                # 使用随机特征作为后备
                batch_size = tf.shape(handcrafted_features)[0]
                deep_features = tf.random.normal((batch_size, self.deep_dim))
        else:
            # 如果没有原始音频，使用随机深度特征作为占位符
            batch_size = tf.shape(handcrafted_features)[0]
            deep_features = tf.random.normal((batch_size, self.deep_dim))

        # 多模态交叉注意力融合
        try:
            if return_attention_scores:
                fused_features, attention_scores = self.mm_caf(
                    [deep_features, handcrafted_features],
                    training=training,
                    return_attention_scores=True
                )
            else:
                fused_features = self.mm_caf(
                    [deep_features, handcrafted_features],
                    training=training
                )
        except Exception as e:
            print(f"Warning: Cross-attention fusion failed: {e}")
            # 简单拼接作为后备
            try:
                deep_proj = layers.Dense(self.embedding_dim)(deep_features)
                hand_proj = layers.Dense(self.embedding_dim)(handcrafted_features)
                fused_features = (deep_proj + hand_proj) / 2
            except Exception as e2:
                print(f"Warning: Fallback fusion also failed: {e2}")
                # 最简单的拼接
                fused_features = tf.concat([deep_features, handcrafted_features], axis=-1)
                fused_features = layers.Dense(self.embedding_dim)(fused_features)

            if return_attention_scores:
                attention_scores = tf.zeros((tf.shape(handcrafted_features)[0], 8))

        # 最终投影
        output = self.final_projection(fused_features, training=training)

        if return_attention_scores:
            return output, attention_scores
        return output


# --- ProtoNet with Uncertainty Weighting ---
# --- 修复后的增强ProtoNet ---
class EnhancedProtoNet(Model):
    """
    增强的原型网络，集成多模态交叉注意力和不确定性加权损失
    """

    def __init__(self, num_classes=2, embedding_dim=512, temperature=1.0,
                 use_uncertainty_weighting=True, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # 多模态编码器
        self.encoder = MultiModalEncoder(embedding_dim=embedding_dim)

        # 温度参数
        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=tf.constant_initializer(temperature),
            trainable=True
        )

        # 不确定性加权损失
        if use_uncertainty_weighting:
            self.uncertainty_loss = UncertaintyWeightedLoss(num_tasks=3)

        # 损失函数
        self.ce_loss = SparseCategoricalCrossentropy(from_logits=True)

    def build(self, input_shape):
        """构建模型"""
        super().build(input_shape)

    def call(self, inputs, training=False, return_attention=False):
        """前向传播"""
        if return_attention:
            embeddings, attention_scores = self.encoder(
                inputs, training=training, return_attention_scores=True
            )
            return embeddings, attention_scores
        else:
            embeddings = self.encoder(inputs, training=training)
            return embeddings

    def compute_prototypes(self, embeddings, labels):
        """计算类原型"""
        prototypes = []
        for class_id in range(self.num_classes):
            mask = tf.equal(labels, class_id)
            class_embeddings = tf.boolean_mask(embeddings, mask)
            if tf.shape(class_embeddings)[0] > 0:
                prototype = tf.reduce_mean(class_embeddings, axis=0)
            else:
                prototype = tf.zeros(self.embedding_dim)
            prototypes.append(prototype)
        return tf.stack(prototypes)

    def compute_distances(self, embeddings, prototypes):
        """计算到原型的距离"""
        # 欧几里得距离
        distances = tf.norm(
            tf.expand_dims(embeddings, 1) - tf.expand_dims(prototypes, 0),
            axis=2
        )
        return -distances / self.temperature

    def supervised_contrastive_loss(self, embeddings, labels, temperature=0.1):
        """监督对比损失"""
        # 归一化嵌入
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)

        # 计算相似度矩阵
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature

        # 创建标签掩码
        labels = tf.expand_dims(labels, 1)
        mask = tf.equal(labels, tf.transpose(labels))
        mask = tf.cast(mask, tf.float32)

        # 移除对角线
        logits_mask = tf.ones_like(mask) - tf.eye(tf.shape(mask)[0])
        mask = mask * logits_mask

        # 计算对比损失
        exp_logits = tf.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8)

        # 避免除零
        mask_sum = tf.reduce_sum(mask, axis=1)
        mask_sum = tf.where(mask_sum > 0, mask_sum, tf.ones_like(mask_sum))

        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / mask_sum
        loss = -tf.reduce_mean(mean_log_prob_pos)

        return loss

    def clinical_center_loss(self, embeddings, labels, updrs_scores, temperature=0.1):
        """临床中心损失"""
        embeddings = tf.cast(embeddings, tf.float32)
        labels = tf.cast(labels, tf.int32)
        updrs_scores = tf.cast(updrs_scores, tf.float32)

        # 计算原型
        prototypes = tf.math.unsorted_segment_mean(embeddings, labels, num_segments=2)

        # 仅对PD类考虑严重度
        mask_pd = tf.cast(labels, tf.bool)
        emb_pd = tf.boolean_mask(embeddings, mask_pd)
        score_pd = tf.boolean_mask(updrs_scores, mask_pd)

        if tf.shape(emb_pd)[0] > 0:
            # 计算到原型距离
            dist = tf.norm(emb_pd - prototypes[1], axis=1)
            # 希望距离与严重度成反比
            max_score = tf.reduce_max(score_pd)
            max_score = tf.where(max_score > 0, max_score, tf.ones_like(max_score))
            target = 1.0 - (score_pd / max_score)
            loss = tf.reduce_mean(tf.square(dist - target))
        else:
            loss = tf.constant(0.0, dtype=tf.float32)

        return loss

    def compute_loss(self, inputs, labels, updrs_scores=None):
        """计算总损失"""
        embeddings = self(inputs, training=True)

        # 计算原型和距离
        prototypes = self.compute_prototypes(embeddings, labels)
        logits = self.compute_distances(embeddings, prototypes)

        # 分类损失
        ce_loss = self.ce_loss(labels, logits)

        # 对比损失
        contrastive_loss = self.supervised_contrastive_loss(embeddings, labels)

        # 临床损失
        if updrs_scores is not None and self.config.get('USE_CLINICAL_CENTER_LOSS', False):
            clinical_loss = self.clinical_center_loss(embeddings, labels, updrs_scores)
        else:
            clinical_loss = tf.constant(0.0, dtype=tf.float32)

        # 确保所有损失都是张量
        losses = [
            tf.convert_to_tensor(ce_loss, dtype=tf.float32),
            tf.convert_to_tensor(contrastive_loss, dtype=tf.float32),
            tf.convert_to_tensor(clinical_loss, dtype=tf.float32)
        ]

        # 不确定性加权
        if self.use_uncertainty_weighting:
            total_loss, weighted_losses, precision_losses = self.uncertainty_loss(losses)
            return {
                'total_loss': total_loss,
                'ce_loss': ce_loss,
                'contrastive_loss': contrastive_loss,
                'clinical_loss': clinical_loss,
                'weighted_losses': weighted_losses,
                'precision_losses': precision_losses,
                'logits': logits
            }
        else:
            # 传统加权
            total_loss = losses[0] + 0.5 * losses[1] + 0.5 * losses[2]
            return {
                'total_loss': total_loss,
                'ce_loss': ce_loss,
                'contrastive_loss': contrastive_loss,
                'clinical_loss': clinical_loss,
                'logits': logits
            }


# --- 特征提取函数 ---
def extract_audio_features(file_path, sr=16000, fast_mode=True):
    """优化的音频特征提取函数"""
    try:
        # 快速模式：只加载前3秒
        if fast_mode:
            y, _ = librosa.load(file_path, sr=sr, duration=3.0)
        else:
            y, _ = librosa.load(file_path, sr=sr)

        if len(y) == 0:
            return None, None

        # 基础特征字典
        features = {}

        # 1. 基频特征 (F0) - 使用更稳定的方法
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
            )
            f0_clean = f0[~np.isnan(f0)]

            if len(f0_clean) > 0:
                features['f0_mean'] = np.mean(f0_clean)
                features['f0_std'] = np.std(f0_clean)
                features['f0_range'] = np.ptp(f0_clean)
                features['f0_median'] = np.median(f0_clean)
            else:
                features['f0_mean'] = features['f0_std'] = features['f0_range'] = features['f0_median'] = 0
        except Exception as e:
            print(f"F0 extraction failed: {e}")
            features['f0_mean'] = features['f0_std'] = features['f0_range'] = features['f0_median'] = 0

        # 2. MFCC特征
        try:
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i + 1}_mean'] = np.mean(mfccs[i])
                features[f'mfcc_{i + 1}_std'] = np.std(mfccs[i])
        except Exception as e:
            print(f"MFCC extraction failed: {e}")
            for i in range(13):
                features[f'mfcc_{i + 1}_mean'] = 0
                features[f'mfcc_{i + 1}_std'] = 0

        # 3. 频谱特征
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            features['spectral_centroid_std'] = np.std(spectral_centroids)

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)

            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
            features['zcr_mean'] = np.mean(zero_crossing_rate)
            features['zcr_std'] = np.std(zero_crossing_rate)
        except Exception as e:
            print(f"Spectral features extraction failed: {e}")
            features['spectral_centroid_mean'] = features['spectral_centroid_std'] = 0
            features['spectral_rolloff_mean'] = features['spectral_rolloff_std'] = 0
            features['zcr_mean'] = features['zcr_std'] = 0

        # 4. 色度特征
        try:
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma)
            features['chroma_std'] = np.std(chroma)
        except Exception as e:
            print(f"Chroma extraction failed: {e}")
            features['chroma_mean'] = features['chroma_std'] = 0

        # 5. 时域统计特征
        try:
            features['rms_energy'] = np.sqrt(np.mean(y ** 2))
            features['kurtosis'] = kurtosis(y)
            features['skewness'] = skew(y)
            features['signal_mean'] = np.mean(y)
            features['signal_std'] = np.std(y)
        except Exception as e:
            print(f"Time domain features extraction failed: {e}")
            features['rms_energy'] = features['kurtosis'] = features['skewness'] = 0
            features['signal_mean'] = features['signal_std'] = 0

        # 6. 抖动和微颤特征（简化版本）
        try:
            if len(f0_clean) > 1:
                jitter = np.std(np.diff(f0_clean)) / np.mean(f0_clean) if np.mean(f0_clean) > 0 else 0
                features['jitter'] = jitter
            else:
                features['jitter'] = 0
        except Exception as e:
            print(f"Jitter extraction failed: {e}")
            features['jitter'] = 0

        return features, y

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


def load_data_from_wav_files_optimized(config):
    """
    从WAV文件加载数据 - 带缓存功能的优化版本
    """
    print('Loading data from WAV files...')

    # 数据路径检查
    data_path = Path(config['DATA_PATH'])
    hc_dir = data_path / 'HC1'
    pd_dir = data_path / 'PD1'

    print(f'📂 数据路径: {data_path}')
    print(f'📂 HC1路径: {hc_dir}')
    print(f'📂 PD1路径: {pd_dir}')

    # 检查目录是否存在
    if not hc_dir.exists():
        print(f'❌ HC1目录不存在: {hc_dir}')
        return None
    if not pd_dir.exists():
        print(f'❌ PD1目录不存在: {pd_dir}')
        return None

    # 缓存文件路径
    cache_dir = data_path / 'cache'
    cache_dir.mkdir(exist_ok=True)

    features_cache_file = cache_dir / 'features_cache.pkl'
    audio_cache_file = cache_dir / 'audio_cache.pkl'

    # 检查是否存在缓存
    if features_cache_file.exists() and audio_cache_file.exists():
        try:
            print('🔄 发现缓存文件，正在加载...')

            # 加载缓存的特征
            with open(features_cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            # 加载缓存的音频数据
            with open(audio_cache_file, 'rb') as f:
                cached_audio = pickle.load(f)

            df = pd.DataFrame(cached_data['features'])
            df[config['TARGET_COLUMN']] = cached_data['labels']

            print(f'✅ 从缓存加载 {len(df)} 条样本，分布:')
            print(df[config['TARGET_COLUMN']].value_counts())
            print(f'特征维度: {len(df.columns) - 1}')

            # 询问是否使用缓存
            use_cache = input('是否使用缓存数据？(y/n，默认y): ').strip().lower()
            if use_cache in ['', 'y', 'yes']:
                return df, cached_audio['audio_data']
            else:
                print('用户选择重新提取特征...')

        except Exception as e:
            print(f'⚠️ 缓存加载失败: {e}，将重新提取特征')

    # 重新提取特征
    print('🔍 开始提取特征...')

    # 搜索WAV文件
    wav_files = list(hc_dir.rglob('*.wav')) + list(pd_dir.rglob('*.wav'))

    if not wav_files:
        print('❌ 未找到任何WAV文件')
        print(f'请检查以下目录中是否包含.wav文件:')
        print(f'  - {hc_dir}')
        print(f'  - {pd_dir}')
        return None

    print(f'📁 共发现 {len(wav_files)} 个 wav 文件')

    # 显示文件分布
    hc_files = [f for f in wav_files if 'HC1' in str(f)]
    pd_files = [f for f in wav_files if 'PD1' in str(f)]
    print(f'   - HC1: {len(hc_files)} 个文件')
    print(f'   - PD1: {len(pd_files)} 个文件')

    # 可选：限制文件数量用于快速调试
    if config.get('MAX_FILES_FOR_DEBUG'):
        wav_files = wav_files[:config['MAX_FILES_FOR_DEBUG']]
        print(f'📁 调试模式：限制为 {len(wav_files)} 个文件')

    all_features, all_labels, all_audio = [], [], []

    for file_path in tqdm(wav_files, desc='提取特征'):
        # 标签逻辑 - 基于路径判断
        if 'PD1' in str(file_path):
            label = 1  # 帕金森病
        elif 'HC1' in str(file_path):
            label = 0  # 健康对照
        else:
            print(f'⚠️ 无法确定文件标签: {file_path}')
            continue

        features, audio = extract_audio_features(str(file_path), config['SAMPLE_RATE'])
        if features is not None:
            all_features.append(features)
            all_labels.append(label)
            all_audio.append(audio)

    if not all_features:
        print('❌ 未提取到有效特征')
        return None

    # 保存到缓存
    try:
        print('💾 保存特征到缓存...')

        # 保存特征缓存
        cache_data = {
            'features': all_features,
            'labels': all_labels,
            'timestamp': datetime.now().isoformat(),
            'config': dict(config),  # 转换为普通字典以便序列化
            'file_count': len(wav_files),
            'hc_count': len([l for l in all_labels if l == 0]),
            'pd_count': len([l for l in all_labels if l == 1])
        }
        with open(features_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

        # 保存音频缓存
        audio_cache_data = {
            'audio_data': all_audio,
            'timestamp': datetime.now().isoformat()
        }
        with open(audio_cache_file, 'wb') as f:
            pickle.dump(audio_cache_data, f)

        print('✅ 缓存保存成功')
        print(f'   - 特征缓存: {features_cache_file}')
        print(f'   - 音频缓存: {audio_cache_file}')

    except Exception as e:
        print(f'⚠️ 缓存保存失败: {e}')

    df = pd.DataFrame(all_features)
    df[config['TARGET_COLUMN']] = all_labels
    print(f'✅ 成功加载 {len(df)} 条样本，分布:')
    print(df[config['TARGET_COLUMN']].value_counts())
    print(f'特征维度: {len(df.columns) - 1}')

    return df, all_audio

# --- 训练和评估函数 ---
class EnhancedTrainer:
    """增强的训练器，支持多模态和不确定性加权"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def prepare_data(self, df, audio_data):
        """优化的数据准备方法"""
        print("Preparing data for training...")

        # 分离特征和标签
        feature_columns = [col for col in df.columns if col != self.config['TARGET_COLUMN']]
        X_handcrafted = df[feature_columns].values.astype(np.float32)
        y = df[self.config['TARGET_COLUMN']].values.astype(np.int32)

        # 处理缺失值
        imputer = SimpleImputer(strategy='median')
        X_handcrafted = imputer.fit_transform(X_handcrafted)

        # 标准化手工特征
        if self.config['SCALE_DATA']:
            X_handcrafted = self.scaler.fit_transform(X_handcrafted)

        # 音频数据预处理 - 大幅减少目标长度以加速处理
        target_length = self.config['SAMPLE_RATE'] * 3  # 改为3秒而不是30秒
        X_audio = np.zeros((len(audio_data), target_length), dtype=np.float32)

        print(f"Processing {len(audio_data)} audio files (target length: {target_length})...")

        for i, audio in enumerate(tqdm(audio_data, desc="Preparing audio")):
            if audio is None or len(audio) == 0:
                continue

            if len(audio) > target_length:
                # 取中间部分而不是开头，通常包含更多信息
                start_idx = (len(audio) - target_length) // 2
                X_audio[i] = audio[start_idx:start_idx + target_length]
            else:
                X_audio[i, :len(audio)] = audio

        print(f"Data preparation completed:")
        print(f"  Handcrafted features: {X_handcrafted.shape}")
        print(f"  Audio data: {X_audio.shape}")
        print(f"  Labels: {y.shape}")

        return X_handcrafted, X_audio, y

    def create_model(self):
        """创建增强的ProtoNet模型"""
        self.model = EnhancedProtoNet(
            num_classes=2,
            embedding_dim=self.config['EMBEDDING_DIM'],
            temperature=1.0,
            use_uncertainty_weighting=self.config['USE_UNCERTAINTY_WEIGHTING']
        )

        return self.model

    # --- 修复后的训练器train_fold方法 ---
    def train_fold(self, X_handcrafted, X_audio, y, train_idx, val_idx, fold):
        """训练单个折"""
        print(f"\nTraining Fold {fold + 1}")

        # 分割数据
        X_train_hc, X_val_hc = X_handcrafted[train_idx], X_handcrafted[val_idx]
        X_train_audio, X_val_audio = X_audio[train_idx], X_audio[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # SMOTE处理（仅对传统特征）
        if self.config['USE_SMOTE'] and SMOTE_AVAILABLE:
            try:
                # 检查最小类别样本数
                unique, counts = np.unique(y_train, return_counts=True)
                min_samples = min(counts)
                k_neighbors = min(self.config['SMOTE_K_NEIGHBORS'], min_samples - 1)

                if k_neighbors > 0:
                    smote = SMOTE(random_state=self.config['RANDOM_STATE'],
                                  k_neighbors=k_neighbors)
                    X_train_hc_smote, y_train_smote = smote.fit_resample(X_train_hc, y_train)

                    # 为SMOTE生成的样本创建对应的音频数据（使用最近邻）
                    from sklearn.neighbors import NearestNeighbors
                    nn = NearestNeighbors(n_neighbors=1)
                    nn.fit(X_train_hc)
                    _, indices = nn.kneighbors(X_train_hc_smote[len(X_train_hc):])

                    X_train_audio_smote = np.vstack([
                        X_train_audio,
                        X_train_audio[indices.flatten()]
                    ])

                    X_train_hc, X_train_audio, y_train = X_train_hc_smote, X_train_audio_smote, y_train_smote
                    print(f"SMOTE applied: {len(y_train)} samples after augmentation")
                else:
                    print("SMOTE skipped: insufficient samples for k_neighbors")
            except Exception as e:
                print(f"SMOTE failed: {e}, continuing without SMOTE")

        # 创建模型
        model = self.create_model()

        # 优化器
        optimizer = AdamW(
            learning_rate=self.config['LEARNING_RATE'],
            weight_decay=self.config['WEIGHT_DECAY']
        )

        # 训练循环
        best_val_acc = 0
        patience_counter = 0
        train_losses = []
        val_accuracies = []

        for epoch in range(self.config['EPOCHS']):
            # 训练步骤
            epoch_losses = []

            # 批次训练
            n_samples = len(X_train_hc)
            indices = np.random.permutation(n_samples)

            for start_idx in range(0, n_samples, self.config['BATCH_SIZE']):
                end_idx = min(start_idx + self.config['BATCH_SIZE'], n_samples)
                batch_indices = indices[start_idx:end_idx]

                batch_hc = X_train_hc[batch_indices]
                batch_audio = X_train_audio[batch_indices]
                batch_y = y_train[batch_indices]

                # 准备输入
                inputs = {
                    'handcrafted': tf.constant(batch_hc, dtype=tf.float32),
                    'raw_audio': tf.constant(batch_audio, dtype=tf.float32)
                }

                # 计算损失和梯度
                with tf.GradientTape() as tape:
                    try:
                        loss_dict = model.compute_loss(inputs, batch_y)
                        total_loss = loss_dict['total_loss']

                        # 确保损失是标量张量
                        if tf.rank(total_loss) > 0:
                            total_loss = tf.reduce_mean(total_loss)

                    except Exception as e:
                        print(f"Warning: Loss computation failed: {e}")
                        # 使用简单的分类损失作为后备
                        try:
                            embeddings = model(inputs, training=True)
                            prototypes = model.compute_prototypes(embeddings, batch_y)
                            logits = model.compute_distances(embeddings, prototypes)
                            total_loss = model.ce_loss(batch_y, logits)
                        except Exception as e2:
                            print(f"Warning: Fallback loss also failed: {e2}")
                            continue

                # 反向传播
                try:
                    gradients = tape.gradient(total_loss, model.trainable_variables)
                    # 过滤None梯度并进行梯度裁剪
                    filtered_gradients = []
                    filtered_variables = []
                    for grad, var in zip(gradients, model.trainable_variables):
                        if grad is not None:
                            grad = tf.clip_by_norm(grad, 1.0)
                            filtered_gradients.append(grad)
                            filtered_variables.append(var)

                    if filtered_gradients:
                        optimizer.apply_gradients(zip(filtered_gradients, filtered_variables))
                        epoch_losses.append(float(total_loss.numpy()))

                except Exception as e:
                    print(f"Warning: Gradient update failed: {e}")
                    continue

            # 验证
            try:
                val_inputs = {
                    'handcrafted': tf.constant(X_val_hc, dtype=tf.float32),
                    'raw_audio': tf.constant(X_val_audio, dtype=tf.float32)
                }

                val_embeddings = model(val_inputs, training=False)
                val_prototypes = model.compute_prototypes(val_embeddings, y_val)
                val_logits = model.compute_distances(val_embeddings, val_prototypes)
                val_preds = tf.argmax(val_logits, axis=1).numpy()
                val_acc = accuracy_score(y_val, val_preds)
            except Exception as e:
                print(f"Warning: Validation failed: {e}")
                val_acc = 0.0

            if epoch_losses:
                train_losses.append(np.mean(epoch_losses))
            else:
                train_losses.append(0.0)
            val_accuracies.append(val_acc)

            # 早停检查
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # 保存最佳模型权重
                try:
                    best_weights = model.get_weights()
                except:
                    best_weights = None
            else:
                patience_counter += 1

            if patience_counter >= self.config['PATIENCE']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}: Train Loss = {train_losses[-1]:.4f}, Val Acc = {val_acc:.4f}")

        # 恢复最佳权重
        if best_weights is not None:
            try:
                model.set_weights(best_weights)
            except:
                print("Warning: Failed to restore best weights")

        # 最终评估
        try:
            val_inputs = {
                'handcrafted': tf.constant(X_val_hc, dtype=tf.float32),
                'raw_audio': tf.constant(X_val_audio, dtype=tf.float32)
            }

            val_embeddings = model(val_inputs, training=False)
            val_prototypes = model.compute_prototypes(val_embeddings, y_val)
            val_logits = model.compute_distances(val_embeddings, val_prototypes)
            val_probs = tf.nn.softmax(val_logits).numpy()
            val_preds = tf.argmax(val_logits, axis=1).numpy()

            # 计算指标
            fold_results = {
                'accuracy': accuracy_score(y_val, val_preds),
                'balanced_accuracy': balanced_accuracy_score(y_val, val_preds),
                'roc_auc': roc_auc_score(y_val, val_probs[:, 1]) if len(np.unique(y_val)) > 1 else 0.5,
                'predictions': val_preds,
                'probabilities': val_probs,
                'true_labels': y_val,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }
        except Exception as e:
            print(f"Warning: Final evaluation failed: {e}")
            fold_results = {
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'roc_auc': 0.5,
                'predictions': np.zeros_like(y_val),
                'probabilities': np.random.rand(len(y_val), 2),
                'true_labels': y_val,
                'train_losses': train_losses,
                'val_accuracies': val_accuracies
            }

        return fold_results, model

    # # 将这个方法替换到EnhancedTrainer类中
    # EnhancedTrainer.train_fold = train_fold

    def cross_validate(self, X_handcrafted, X_audio, y):
        """执行交叉验证"""
        skf = StratifiedKFold(n_splits=self.config['N_SPLITS'],
                              shuffle=True,
                              random_state=self.config['RANDOM_STATE'])

        fold_results = []
        all_predictions = []
        all_probabilities = []
        all_true_labels = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_handcrafted, y)):
            try:
                fold_result, model = self.train_fold(X_handcrafted, X_audio, y, train_idx, val_idx, fold)
                fold_results.append(fold_result)

                all_predictions.extend(fold_result['predictions'])
                all_probabilities.extend(fold_result['probabilities'])
                all_true_labels.extend(fold_result['true_labels'])

                print(f"Fold {fold + 1} completed: Accuracy = {fold_result['accuracy']:.4f}")
            except Exception as e:
                print(f"Fold {fold + 1} failed: {e}")
                continue

        if not fold_results:
            print("All folds failed!")
            return None

        # 汇总结果
        self.results = {
            'fold_results': fold_results,
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_balanced_accuracy': np.mean([r['balanced_accuracy'] for r in fold_results]),
            'std_balanced_accuracy': np.std([r['balanced_accuracy'] for r in fold_results]),
            'mean_roc_auc': np.mean([r['roc_auc'] for r in fold_results]),
            'std_roc_auc': np.std([r['roc_auc'] for r in fold_results]),
            'all_predictions': np.array(all_predictions),
            'all_probabilities': np.array(all_probabilities),
            'all_true_labels': np.array(all_true_labels)
        }

        return self.results


def manage_cache(config):
    """
    缓存管理功能
    """
    data_path = Path(config['DATA_PATH'])
    cache_dir = data_path / 'cache'

    print(f'📂 缓存目录: {cache_dir}')

    if not cache_dir.exists():
        print('📂 缓存目录不存在，将在首次运行时创建')
        return True  # 继续执行，让程序创建缓存

    features_cache_file = cache_dir / 'features_cache.pkl'
    audio_cache_file = cache_dir / 'audio_cache.pkl'

    print('\n=== 缓存管理 ===')

    # 显示缓存信息
    if features_cache_file.exists():
        try:
            with open(features_cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            print(f'📊 特征缓存信息:')
            print(f'   - 样本数量: {len(cached_data["features"])}')
            print(f'   - HC样本: {cached_data.get("hc_count", "未知")}')
            print(f'   - PD样本: {cached_data.get("pd_count", "未知")}')
            print(f'   - 创建时间: {cached_data.get("timestamp", "未知")}')
            print(f'   - 文件大小: {features_cache_file.stat().st_size / 1024 / 1024:.2f} MB')

            if audio_cache_file.exists():
                print(f'   - 音频缓存大小: {audio_cache_file.stat().st_size / 1024 / 1024:.2f} MB')

        except Exception as e:
            print(f'❌ 缓存信息读取失败: {e}')
    else:
        print('📂 未找到特征缓存文件')

    # 缓存操作选项
    print('\n缓存操作选项:')
    print('1. 使用现有缓存 (推荐)')
    print('2. 清除缓存并重新提取')
    print('3. 查看缓存详细信息')
    print('4. 继续使用缓存')

    choice = input('请选择操作 (1-4, 默认1): ').strip()

    if choice == '2':
        # 清除缓存
        try:
            if features_cache_file.exists():
                features_cache_file.unlink()
                print(f'🗑️ 已删除特征缓存: {features_cache_file}')
            if audio_cache_file.exists():
                audio_cache_file.unlink()
                print(f'🗑️ 已删除音频缓存: {audio_cache_file}')
            print('✅ 缓存已清除，将重新提取特征')
            return False  # 不使用缓存
        except Exception as e:
            print(f'❌ 缓存清除失败: {e}')
            return False

    elif choice == '3':
        # 查看详细信息
        if features_cache_file.exists():
            try:
                with open(features_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)

                print(f'\n📋 缓存详细信息:')
                print(f'   - 特征数量: {len(cached_data["features"])}')
                print(f'   - 标签分布: {pd.Series(cached_data["labels"]).value_counts().to_dict()}')

                # 显示特征名称
                if cached_data["features"]:
                    feature_names = list(cached_data["features"][0].keys())
                    print(f'   - 特征维度: {len(feature_names)}')
                    print(f'   - 前10个特征: {feature_names[:10]}')

            except Exception as e:
                print(f'❌ 详细信息读取失败: {e}')

        return True  # 使用缓存

    else:
        # 默认使用缓存
        return True


# --- 主函数 ---
def main():
    """主函数"""
    print("Enhanced Parkinson's Disease Voice Analysis with Multi-Modal Cross-Attention Fusion")
    print("=" * 80)

    # 创建输出目录
    output_path = Path(CONFIG['OUTPUT_PATH'])
    output_path.mkdir(exist_ok=True)

    try:
        """主函数 - 集成缓存管理"""
        print("Enhanced Parkinson's Disease Voice Analysis with Multi-Modal Cross-Attention Fusion")
        print("=" * 80)

        # 1. 缓存管理
        print("\n0. 缓存管理...")
        use_cache = manage_cache(CONFIG)

        # 2. 加载和预处理数据
        print("\n1. Loading and preprocessing data...")

        if use_cache:
            # 尝试从缓存加载
            result = load_data_from_wav_files_optimized(CONFIG)
        else:
            # 强制重新提取
            cache_dir = Path(CONFIG['DATA_PATH']) / 'cache'
            features_cache_file = cache_dir / 'features_cache.pkl'
            audio_cache_file = cache_dir / 'audio_cache.pkl'

            # 临时删除缓存文件以强制重新提取
            if features_cache_file.exists():
                features_cache_file.unlink()
            if audio_cache_file.exists():
                audio_cache_file.unlink()

            result = load_data_from_wav_files_optimized(CONFIG)

        if result is None:
            print("❌ 数据加载失败")
            return

        df, audio_data = result
        print(f"Loaded {len(df)} samples with {len(df.columns) - 1} features")

        # 3. 准备数据
        print("\n2. Preparing data for training...")
        trainer = EnhancedTrainer(CONFIG)
        X_handcrafted, X_audio, y = trainer.prepare_data(df, audio_data)

        print(f"Handcrafted features shape: {X_handcrafted.shape}")
        print(f"Audio data shape: {X_audio.shape}")
        print(f"Labels shape: {y.shape}")

        # 4. 训练和评估
        print("\n3. Training and evaluating model...")
        results = trainer.cross_validate(X_handcrafted, X_audio, y)

        # 5. 生成可视化
        print("\n5. Generating visualizations...")
        visualizer = SCIPaperVisualization(results, CONFIG['OUTPUT_PATH'])
        visualizer.create_all_figures()

        print("\nAnalysis completed successfully!")
        print(f"Results and visualizations saved to: {output_path}")

    except Exception as e:
        print(f"\nError during execution: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
