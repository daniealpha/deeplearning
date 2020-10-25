import tensorflow as tf
import pandas
import numpy
import os

class DeepFM(tf.keras.Model):
    def __init__(self,
                 num_field, # 原始特征数，离散特征 dim: f
                 num_feature, # 离散特征展开后的总特征数， dim: n
                 embedding_size=32, # embedding 维度， dim: k
                 layer_sizes=[128, 64, 32],
                 dropout_fm=0,
                 dropout_deep=0):
        super().__init__()
        # 参数
        self.num_field = num_field
        self.num_feature = num_feature
        self.embedding_size = embedding_size
        self.layer_sizes = layer_sizes

        self.dropout_fm = dropout_fm
        self.dropout_deep = dropout_deep

        # 初始化FM一阶系数，可以理解为是1维的embedding
        self.first_weights = tf.keras.layers.Embedding(num_feature,
                                                       1,
                                                       embeddings_initializer='normal')

        # 初始化FM一阶系数，可以理解为是k维的embedding
        self.feature_embeddings = tf.keras.layers.Embedding(num_feature,
                                                            embedding_size,
                                                            embeddings_initializer='normal')

        # deep网络结构
        # setattr(x, 'y', v)等价于 x.y = v
        # 网络层个模块的顺序: dense + batchNorm + activation + dropout
        # batchNorm 是为了防止sigmoid/tanh激活函数，这种数据分布在偏向2边的地方梯度消失，所以bn刚开始加在激活前面
        # 而relu没有这种局限，所以bn也可以加在激活之后，业界实验加在激活之后会好些。
        for i in range(len(layer_sizes)):
            setattr(self, 'dense_' + str(i), tf.keras.layers.Dense(layer_sizes[i]))
            setattr(self, 'bn_' + str(i), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i), tf.keras.layers.Activation('relu'))
            # setattr(self, 'activation_' + str(i), tf.keras.layers.ReLU())
            setattr(self, 'dropout_' + str(i), tf.keras.layers.Dropout(dropout_deep))

        # 最后一层全连接

        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    # 前向传播
    @tf.function
    def call(self, inputs):
        feature_index, feature_value = inputs
        # 为了矩阵运算，需要对feature_value进行从内升维，从none * f 到 none * f * 1
        # 之所以是f不是one-hot后的n,是因为这里进行了压缩，离散的值都是1，主要是通过feature_index索引决定的
        feature_value = tf.expand_dims(feature_value, axis=-1)

        # step1: FM 一阶部分 none * f * 1
        first_weights = self.first_weights(feature_index)
        # Wi*Xi, tf.math.multiply 是对应元素相乘,  none * f * 1
        first_weight_value =tf.math.multiply(first_weights, feature_value)

        # first_weight_value维度从none * f * 1降维为 none * f
        #只是降维，可以理解为concat(w1*x1, ..., wn*xn), 并不是sum(w1*x1, ..., wn*xn)
        fm_first_output = tf.math.reduce_sum(first_weight_value, axis=2) # none * f
        fm_first_output = tf.keras.layers.Dropout(self.dropout_fm)(fm_first_output) # none * f

        # step2: FM 二阶部分 none * f * k
        feature_emb = self.feature_embeddings(feature_index) # none * f * k
        # Vi*Xi。tf.math.multiply广播(none * f * k) * (none * f * 1) = (none * f * k)
        feature_emb_value = tf.math.multiply(feature_emb, feature_value) # none * f * k

        # sum(Vi*Xi)**2, none * k
        sumed_feature_emb = tf.math.reduce_sum(feature_emb_value, axis=1) # none * k
        interaction_part1 = tf.math.pow(sumed_feature_emb, 2) # none * k

        # sum((Vi*Xi)**2), none * k
        squared_feature_emb = tf.math.pow(feature_emb_value, 2) # none * f * k
        interaction_part2 = tf.math.reduce_sum(squared_feature_emb, axis=1) # none * k

        # sum(Vi*Vj*Xi*Xj) none * k
        fm_second_output = 0.5 * tf.math.subtract(interaction_part1, interaction_part2) # none * k
        fm_second_output = tf.keras.layers.Dropout(self.dropout_deep)(fm_second_output) # none * k

        # step3: deep部分
        # 输入形状变换为 none * (f * k)
        deep_feature = tf.reshape(feature_emb_value, (-1, self.num_field * self.embedding_size))
        deep_feature = tf.keras.layers.Dropout(self.dropout_deep)(deep_feature)

        # 最终输出：none * layer_sizes[-1]
        for i in range(len(self.layer_sizes)):
            deep_feature = getattr(self, 'dense_' + str(i))(deep_feature)
            deep_feature = getattr(self, 'bn_' + str(i))(deep_feature)
            deep_feature = getattr(self, 'activation_' + str(i))(deep_feature)
            deep_feature = getattr(self, 'dropout_' + str(i))(deep_feature)

        # 所有中间特征合并：none * (f + k + layer_sizes[-1])
        concat_input = tf.concat((fm_first_output, fm_second_output, deep_feature), axis=1)

        # 最终输出
        output = self.output_layer(concat_input)

        return output