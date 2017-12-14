import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_integer('batchSize', 10, "Number of samples per batch.")

class Model(object):
    def __init__(self, modelName, inputData):
        self.modelName = modelName
        # x = tf.placeholder("float32", shape=inputShape)
        self.outputs = [inputData]
    
    def _getLayerName(self, layerNum=None):
        if layerNum is None:
            layerNum = len(self.outputs)
        return '%s_%03d' % (self.modelName, layerNum+1)

    def _getInputNum(self):
        return int(self.getOutput().get_shape()[-1])
    
    # This function was quoted from https://github.com/david-gpu/srez/blob/master/srez_model.py
    def _glorot_initializer_conv2d(self, prev_units, num_units, mapsize, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.
        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
        stddev  = np.sqrt(stddev_factor / (np.sqrt(prev_units*num_units)*mapsize*mapsize))
        return tf.truncated_normal([mapsize, mapsize, prev_units, num_units], mean=0.0, stddev=stddev)

    def getOutput(self):
        return self.outputs[-1]
    
    def addBatchNorm(self, scale=False):
        with tf.variable_scope(self._getLayerName()):
            out = tf.contrib.layers.batch_norm(self.getOutput(), scale=scale)
        
        self.outputs.append(out)
        return self

    def addSigmoid(self):
        with tf.variable_scope(self._getLayerName()):
            out = tf.nn.sigmoid(self.getOutput())
        self.outputs.append(out)
        return self
    
    def addLeakyRelu(self, param=0.2):
        with tf.variable_scope(self._getLayerName()):
            val1 = 0.5 * (1 + param)
            val2 = 0.5 * (1 - param)
            out = val1 * self.getOutput() + val2 * tf.abs(self.getOutput())
        self.outputs.append(out)
        return self
    
    def addReduceMean(self):
        with tf.variable_scope(self._getLayerName()):
            out = tf.reduce_mean(self.getOutput())
        self.outputs.append(out)
        return self

    def addSigmoid(self):
        with tf.variable_scope(self._getLayerName()):
            out = tf.nn.sigmoid(self.getOutput())

        self.outputs.append(out)
        return self

    def addConv2d(self, outputUnits, mapsize=1, stride=1, stddevFactor=1.0):
        with tf.variable_scope(self._getLayerName()):
            inputUnits = self._getInputNum()
            
            # add weight conv
            elementsW = self._glorot_initializer_conv2d(inputUnits, outputUnits, mapsize, stddev_factor=stddevFactor)
            weight = tf.get_variable('weight', initializer=elementsW)
            out = tf.nn.conv2d(self.getOutput(), weight, strides=[1, stride, stride, 1], padding='SAME')
            
            # add bias
            elementsB = tf.constant(0.0, shape=[outputUnits])
            bias = tf.get_variable('bias', initializer=elementsB)
            out = tf.nn.bias_add(out, bias)
        
        self.outputs.append(out)
        return self
    
    def addConv2dTranspose(self, outputUnits, mapsize=1, stride=1, stddevFactor=1.0):
        with tf.variable_scope(self._getLayerName()):
            inputUnits = self._getInputNum()
            
            # add weight conv
            elementsW = self._glorot_initializer_conv2d(inputUnits, outputUnits, mapsize, stddev_factor=stddevFactor)
            weight = tf.get_variable('weight', initializer=elementsW)
            weight = tf.transpose(weight, perm=[0, 1, 3, 2])
            prevOutput = self.getOutput()
            #outputShape = [FLAGS.batchSize,
            outputShape = [int(prevOutput.get_shape()[0]),
                           int(prevOutput.get_shape()[1]) * stride,
                           int(prevOutput.get_shape()[2]) * stride,
                           outputUnits]
            out = tf.nn.conv2d_transpose(self.getOutput(), weight,
                                         output_shape=outputShape,
                                         strides=[1, stride, stride, 1],
                                         padding='SAME')

            # add bias
            elementsB = tf.constant(0.0, shape=[outputUnits])
            bias = tf.get_variable('bias', initializer=elementsB)
            out = tf.nn.bias_add(out, bias)
        
        self.outputs.append(out)
        return self

def _generator(inputData):
    # ex: inputShape = [FLAGS.batchSize, 512, 512, 3]
    generatorModel = Model("generator", inputData)
    oldVars = tf.all_variables()

    # conv params
    convUnits = [64, 128, 256, 512]
    mapsize = 2
    stride = 2
    miniCnt = 0

    # conv layers
    for unit in convUnits:
        if not generatorModel.getOutput().get_shape()[1] % mapsize == 0:
            stride = 5
        else:
            miniCnt += 1
        generatorModel.addConv2d(unit, mapsize=mapsize, stride=stride, stddevFactor=1.0)
        generatorModel.addLeakyRelu(param=0.1)
        print(generatorModel.getOutput().get_shape())
  
    print("Trans")
    convTransUnits = [1024, 512, 256, 128, 64]
    stride = 5
    # conv transpose layers
    for i, unit in enumerate(convTransUnits):
        if not i < miniCnt:
            stride = 2
        generatorModel.addBatchNorm()
        generatorModel.addConv2dTranspose(unit, mapsize=mapsize, stride=stride, stddevFactor=1.0)
        generatorModel.addLeakyRelu(param=0.2)
        print(generatorModel.getOutput().get_shape())
    
    # last layer
    stride = 2
    generatorModel.addConv2d(3, mapsize=mapsize, stride=stride, stddevFactor=1.0)
    generatorModel.addSigmoid()
    print(generatorModel.getOutput().get_shape())

    newVars = tf.all_variables()

    genVars = list(set(newVars) - set(oldVars))

    return generatorModel.getOutput(), genVars

def _discriminator(inputData):
    discriminatorModel = Model("discriminator", inputData)
    
    oldVars = tf.all_variables()

    # conv params
    convUnits = [32, 64, 128, 256, 512, 1024]
    mapsize = 2
    stride = 2
    
    # conv layers
    for unit in convUnits:
        discriminatorModel.addConv2d(unit, mapsize=mapsize, stride=stride, stddevFactor=1.0)
        discriminatorModel.addLeakyRelu(param=0.2)

    discriminatorModel.addReduceMean()
    discriminatorModel.addSigmoid()
    
    newVars = tf.all_variables()
    
    disVars = list(set(newVars)-set(oldVars))

    return discriminatorModel.getOutput(), disVars

def createModels(inputData, realData, testInputData):
    # real input for discriminator
    # disRealInput = tf.identity(realData, name="disRealInput")
    disRealInput = realData

    # generator
    with tf.variable_scope("gen") as scope:
        genModel, genVars = _generator(inputData)
        scope.reuse_variables()
        genModelTest, _ = _generator(testInputData)


    # discriminator with real data
    with tf.variable_scope("dis") as scope:
        # Instead of output diff loss, this only give discriminator it.
        disModelReal, disVars = _discriminator(tf.sub(disRealInput, inputData))
        scope.reuse_variables()
        disModelFake, _ = _discriminator(tf.sub(genModel, inputData))
        

    return genModel, disModelReal, disModelFake, genVars, disVars

def createGenLoss(disOutput):
    # This loss function use only discriminator output,
    # but discriminator receive difference of generated image and real image.
    # This logic is hoped to create same image as input one.

    crossEntropy = tf.nn.sigmoid_cross_entropy_with_logits(disOutput, tf.ones_like(disOutput))
    genLoss = tf.reduce_mean(crossEntropy, name="genLoss")
    return genLoss

def createDisLoss(disOutputTrue, disOutputFake):
    
    crossEntropyTrue = tf.nn.sigmoid_cross_entropy_with_logits(disOutputTrue, tf.ones_like(disOutputTrue))
    disLossTrue = tf.reduce_mean(crossEntropyTrue, name="disLossTrue")

    crossEntropyFake = tf.nn.sigmoid_cross_entropy_with_logits(disOutputFake, tf.zeros_like(disOutputFake))
    dissLossFake = tf.reduce_mean(crossEntropyFake, name="disLossFake")

    return disLossTrue, disLossFake
    
def createOptimizers(genLoss, disLoss, genVars, disVars):
    globalStep = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
    lr = tf.placeholder(dtype=tf.float32, name="learningRate")
    
    genOpt = tf.train.AdamOptimizer(learning_rate=lr, name="genOptimizer").minimize(genLoss, var_list=genVars, global_step=globalStep)
    disOpt = tf.train.AdamOptimizer(learning_rate=lr, name="disOptimizer").minimize(disLoss, var_list=disVars, global_step=globalStep)
    
    return genOpt, disOpt

# For testing
real = tf.constant(0.1, shape=(183, 500, 500, 3))
test = tf.constant(0.3, shape=(10, 500, 500, 3))
print(createModels(real, real, test))
