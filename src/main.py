import tensorflow as tf
import model
import train
import inputs

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('run', 'train', "choose a mode")
tf.app.flags.DEFINE_string('mosaicImageDir', '../Dataset/MosaicImages/', "Directory that includes mosaic images path")
tf.app.flags.DEFINE_string('labelImageDir', '../Dataset/LabelImages/', "Directory that includes label images path")

tf.app.flags.DEFINE_string('testMosaicDir', '../Dataset/TestMosaic/', "Directory that includes mosaic test images path")
tf.app.flags.DEFINE_string('testLabelDir', '../Dataset/TestLabel/', "Directory that includes label test images path")

tf.app.flags.DEFINE_integer('batchSize', 4, "Number of sample images")
tf.app.flags.DEFINE_float('lrStart', 0.0002, "First learning rate")
tf.app.flags.DEFINE_integer('lrHalfPointEp', 5000, "Number of epoch which changes learning rate")
tf.app.flags.DEFINE_integer('displaySpan', 5, "The span which is displayed some training data")
tf.app.flags.DEFINE_integer('saveSpan', 10, "The span which is saved training data")
tf.app.flags.DEFINE_integer('epochLimit', 50000, "Epoch limit")

def training():
    labelImages, mosaicImages = inputs.setupInputdata(FLAGS.mosaicImageDir, FLAGS.labelImageDir)
    testLabel, testMosaic = inputs.setupInputdata(FLAGS.testMosaicDir, FLAGS.testLabelDir)
    train.trainModels(labelImages, mosaicImages, testLabel, testMosaic)
    

def main(argv=None):
    if FLAGS.run == 'train':
        print("Train Mode!")
        training()

if __name__ == '__main__':
    tf.app.run()
