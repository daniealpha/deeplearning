from util.v1.utils import get_index_dict, get_feature_index_value, train_model
from model.v1.deepFM import DeepFM
import tensorflow as tf
import pandas as pd

def main():
    print('get data...')
    train_data_path = './data/train.csv'
    test_data_path = './data/test.csv'
    dfTrain = pd.read_csv(train_data_path)
    dfTest = pd.read_csv(test_data_path)

    #获取 类别变量/数值变量 名称
    cols = [c for c in dfTrain.columns if c not in ['id', 'target']]
    is_ctg = pd.Series(cols).apply(lambda x:x[-4:]) == '_cat'

    ctg_cols = list(pd.Series(cols)[is_ctg])
    num_cols = list(set(cols) - set(ctg_cols))

    # 获取feature_index的mapping dict
    # 注意要用合并train/test data后来生成dict，以防数据分布不一致
    x_train_test = pd.concat([dfTrain[cols], dfTest[cols]], axis=0)

    # 获取field_num, feature_num, index_dict
    field_num = len(cols)
    feature_num, index_dict = get_index_dict(data=x_train_test,
                                             ctg_cols=ctg_cols,
                                             num_cols=num_cols)

    # 获取feature_index, feature_value
    feature_index, feature_value = get_feature_index_value(data=dfTrain,
                                                           index_dict=index_dict,
                                                           ctg_cols=ctg_cols,
                                                           num_cols=num_cols)

    # 训练的train data & train label, 并统一数据格式，不然tf.math.multiply会报错
    # TypeError: Input 'y' of 'Mul' Op has type float64 that does not match type float32 of argument 'x'.
    feature_index = tf.cast(feature_index, tf.float32)
    feature_value = tf.cast(feature_value, tf.float32)
    y_label = tf.cast(dfTrain['target'].values, tf.float32)

    print('train model...')
    # 初始化模型
    model_deepfm = DeepFM(num_field=field_num,
                          num_feature=feature_num,
                          layer_sizes=[128, 64, 32],
                          embedding_size=32,
                          dropout_fm=0,
                          dropout_deep=0)

    train_model(model_deepfm,
                idx=feature_index,
                value=feature_value,
                label=y_label,
                batch_size=64,
                epochs=5)

    print(model_deepfm.summary())

    print('predict...')
    # model_deepfm.predict报错，待研究。ValueError: Unknown graph. Aborting.
    # 直接调用model(x)或者用model.call(x)进行预测

    # 改成model(x1, x2)格式的输入，模型可以训练并预测，但是没法进行save保存
    model_deepfm(feature_index, feature_value)
    # model_deepfm.call(feature_index, feature_value)

    print('save model...')
    # 双输入时可以训练和预测，但是无法保存模型
    # TypeError: call() missing 1 required positional argument: 'feature_value'
    model_deepfm.save('./output/deepfm/', save_format='tf')

    print('load model and predict...')
    model_load = tf.saved_model.load('./output/deepfm/')
    # <简明的tf2> p81 提到 saved_model后的模型没法用model(x) 进行预测，要用model.call(x)进行显性预测
    model_load.call(feature_index, feature_value)
    print('end!')

if __name__ == '__main__':
    main()




