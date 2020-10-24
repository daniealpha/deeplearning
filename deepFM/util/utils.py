import tensorflow as tf
import math

# data process use

# 从原始数据转化对应的feature_index字典
def get_index_dict(data, ctg_cols, num_cols):
    '''
    :param data: 原始数据
    :param ctg_cols: list 类别特征名称
    :param num_cols: list 数值特征名称
    :return: feature_index的字典 index_dict
    '''
    cols = list(set(ctg_cols) | set(num_cols))
    # 判断两种变量是否有交叉
    if list(set(ctg_cols) & set(num_cols)) != []:
        raise Exception('There has col both in ctg_cols and num_cols!')
        return None

    # 获取字典
    index_dict = {}
    idx = 0
    for col in cols:
        if col in num_cols: # 数值变量处理
            index_dict[col] = idx
            idx += 1
        else: # 类别变量处理
            col_i_value = sorted(list(set(data[col]))) # 获取类别变量的值
            col_i_dict = dict(zip(col_i_value, range(idx, idx + len(col_i_value)))) # 对应的值mapping
            index_dict[col] = col_i_dict
            idx += len(col_i_value)
    num_feature = idx
    return num_feature, index_dict


# 把原始数据转化为 feature_index, feature_value
def get_feature_index_value(data, index_dict, ctg_cols, num_cols):
    '''
    :param data: 原始数据
    :param index_dict: dict 从get_index_dict中返回的index_dict
    :param ctg_cols: list 类别特征名称
    :param num_cols: list 数值特征名称
    :return: 特征索引feature_index, 特征值feature_value
    '''
    cols = list(set(ctg_cols) | set(num_cols))
    data_index = data[cols]
    data_value = data[cols]
    for col in cols:
        # 类别变量的处理
        if col in ctg_cols:
            data_index.loc[:, col] = data_index[col].map(index_dict[col])
            data_value.loc[:, col] = 1
        # 数值变量的处理
        else:
            data_index.loc[:, col] = index_dict[col]
            # data_value[col] = data_value[col]

    index = data_index[cols].values
    value = data_value[cols].values

    return index, value

# label = tf.constant([[0], [1], [1], [0]])
# idx = tf.constant([[1], [2], [3], [4]])
# value = tf.constant([[4], [5], [6], [7]])
# 把数据分为batch
def get_batch_dataset(idx, value, label, batch_size=64):
    '''
    :param idx: array 从get_feature_index_value得到的data_index
    :param value: array 从get_feature_index_value得到的data_value
    :param label: array 样本标签
    :param batch_size: int
    :return: Dataset object
    '''

    # idx = tf.data.Dataset.from_tensor_slices(idx)
    # value = tf.data.Dataset.from_tensor_slices(value)
    # label = tf.data.Dataset.from_tensor_slices(label)

    # batch_dataset = tf.data.Dataset.zip((label, idx, value))
    # batch_dataset = batch_dataset.shuffle(buffer_size=batch_size)
    batch_dataset = tf.data.Dataset.from_tensor_slices((label, idx, value))
    batch_dataset = batch_dataset.batch(batch_size)
    # batch_dataset = batch_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return batch_dataset


# train model use
@tf.function
def cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.losses.binary_crossentropy(y_true, y_pred))

@tf.function
def train_one_step(model, optimizer, idx, value, label):
    with tf.GradientTape() as tape:
        output = model(inputs=[idx, value])
        loss = cross_entropy_loss(y_true=label, y_pred=output)

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 100) for g in grads]
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    return loss

# 训练一个epoch
def train_one_epoch(model, train_batch_dataset, optimizer, epoch, batch_size, train_item_count):
    for batch_idx, (label, idx, value) in enumerate(train_batch_dataset):
        if len(label) == 0:
            break

        loss = train_one_step(model, optimizer, idx, value, label)

        # 每训练100个batch, print一次信息
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{} / {} ({:.2f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(idx), train_item_count,
                100. * batch_idx / math.ceil(int(train_item_count / batch_size)), loss.numpy()))

        batch_idx_i = batch_idx
        batch_size = len(idx)
    # 当样本不能被batch_size整除时，每一轮epoch结束print一次信息
    if (batch_idx_i * batch_size) != train_item_count:
        print('Train Epoch: {} [{} / {} ({:.2f}%)]\tLoss:{:.6f}'.format(
            epoch, train_item_count, train_item_count,
                   100, loss.numpy()))

# train model
def train_model(model,
                idx,
                value,
                label,
                batch_size=64,
                epochs=5):

    train_batch_dataset = get_batch_dataset(idx,
                                            value,
                                            label,
                                            batch_size=batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    for epoch in range(epochs):
        train_one_epoch(model, train_batch_dataset, optimizer, epoch,
              batch_size=batch_size, train_item_count=label.shape[0])
