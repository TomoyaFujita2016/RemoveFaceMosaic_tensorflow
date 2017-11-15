import tensorflow as tf
import model

FLAGS = tf.app.flags.FLAGS

def trainModels():
    sess = tf.Session()
    inputShape = [FLAGS.batchSize, 512, 512, 3]

    genModel, disModel = model.createModels(inputShape)

    genOptimizer, disOptimizer = model.createOptimizers(model.createGenLoss(disModel), model.createDisLoss(disModel))
    
    init = tf.global_variables_initializer()
    sess.run(init)
    lr = FLAGS.lrStart
    for epoch in range(FLAGS.epochLimit):
        sess.run(genOptimizer, feed_dict={x: , labels: , learningRate: lr})



trainModels()
