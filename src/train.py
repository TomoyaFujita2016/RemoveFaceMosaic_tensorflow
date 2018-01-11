import tensorflow as tf
import model

FLAGS = tf.app.flags.FLAGS

def trainModels(labelImages, mosaicImages, testLabel, testMosaic):
    sess = tf.Session()
    #saver = tf.train.Saver()

    genModel, genTestModel, disModelReal, disModelFake, genVars, disVars = \
        model.createModels(mosaicImages, labelImages, mosaicImages)
    
    genLossFake = model.createGenLoss(disModelFake)
    disLossReal, disLossFake = model.createDisLoss(disModelReal, disModelFake)
    disLoss = tf.add(disLossReal, disLossFake)

    genOptimizer, disOptimizer = model.createOptimizers(genLossFake, disLoss, genVars, disVars)
    
    init = tf.initialize_all_variables()
    sess.run(init)

    lr = FLAGS.lrStart
    for epoch in range(FLAGS.epochLimit):
        _, _, gLossVal, dLossRVal, dLossFVal = \
            sess.run([genOptimizer,disOptimizer, genLossFake, disLossReal, disLossFake] )
        if epoch % FLAGS.displaySpan == 0:
            print("Epoch:[" + str(epoch) + " / " + str(FLAGS.epochLimit) + "], GenLoss:[" + str(gLossVal) + "], DisLossR:["+ str(dLossRVal) + "], DisLossF:[" + str(dLossFVal) + "]")
            #print("Epoch:[%3d / %3d], GenLoss:[%3.3f], DisLossR:[%3.3f], DisLossF:[%3.3f]") % (epoch, FLAGS.epochLimit, gLossVal, dLossRVal, dLossFVal)
        if epoch % FLAGS.lrHalfPointEp == 0:
            lr *= 0.5
        #if epoch % FLAGS.saveSpan == 0:
            #saver.save(sess, "../checkpoints/model.ckpt", global_step=epoch)
