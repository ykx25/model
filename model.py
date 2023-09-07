'''
Author: yinkexin yinkexin@jxcc.com
Date: 2023-09-06 09:45:36
LastEditors: yinkexin yinkexin@jxcc.com
LastEditTime: 2023-09-06 14:07:21
FilePath: \test\model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import tensorflow as tf

def create_model():
    # 1.获得100个随机数据集 y = 0.8x + 0.7
    # 随机数 X 是 二维矩阵 [100,1] x 0.8 + 0.7
    with tf.variable_scope('get_data'):
        X = tf.random_normal([100, 1], mean=0.0, stddev=1.0, name='get_data_X')
        # 进行矩阵运算,0.8 需要是二维数组
        y_true = tf.matmul(X, [[0.8]]) + [[0.7]]
    # 2.建立回归模型
    # 模型是 [100,1] * [1,1] + [1,1]
    # 两个参数 (W,0.8) (D,0.7)
    with tf.variable_scope('create_model'):
        W = tf.Variable(initial_value=(tf.random_normal([1, 1])), name='w')
        B = tf.Variable(initial_value=(tf.random_normal([1, 1])), name='b')
        # 进行矩阵运算
        y_predict = tf.matmul(X, W) + B
    return W, B, y_predict

def test():
    # 3.确定误差
    with tf.variable_scope('get_loss'):
        loss = tf.reduce_mean(tf.square(y_predict - y_true))
    # 4.梯度训练学习，减小误差
    with tf.variable_scope('study'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
 
    init = tf.global_variables_initializer() # 初始化变量
 
    with tf.Session() as sess: # 创建会话
        sess.run(init) #会话执行初始化
        for i in range(1000): # 循环执行1000次训练
            sess.run(optimizer) # 会话执行梯度训练
            wid, bias, loss1 = sess.run([W, B, loss])
            print('\n第%s次训练，本次权重%s' % (i, wid))
            print('第%s次训练，本次偏置%s' % (i, bias))
            print('第%s次训练，本次误差%s\n' % (i, loss1))
            if loss1 <= 0.00000001: # 如果误差小于这个数值，则停止
                 return
        # 保存模型
        model = tf.train.Saver()
        model.save('model.ckpt', save_format='tf')

if __name__ == '__main__':
    W, B, y_predict = create_model()
    test()