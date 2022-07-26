InputSize = 64
HiddenSize = 64
NumChannels = 2
BatchSize = 64

#Mp3BitRate = '128k'
#Mp3BitRate = 'techno'
Mp3BitRate = '128k-v2'
#Mp3BitRate = '128k-v3'

ModelFileName = "./models/" + f'BestModel-{Mp3BitRate}-{InputSize}-{HiddenSize}.h5'
DATAPATH = './training-data/'


LearningRate = 1e-4
Epochs = 100