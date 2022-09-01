scope_table = {
    "resnet_18":               {"pool1":64,   "pool2": 64,  "pool3": 256,  "pool4": 512,"out4":2048},
    "resnet50":               {"pool1":64,   "pool2": 64,  "pool3": 256,  "pool4": 512,"out4":2048},
    "resnet_101":              {"pool1":64,   "pool2": 64,  "pool3": 256,  "pool4": 512,"out4":2048},
    "resnet_152":              {"pool1":64,   "pool2": 64,  "pool3": 256,  "pool4": 512,"out4":2048},
    "resnext50_32x4d":         {"pool1":64,   "pool2": 64,  "pool3": 256,  "pool4": 512,"out4":2048},
    "resnext101_32x8d":        {"pool1":64,   "pool2": 64,  "pool3": 256,  "pool4": 512,"out4":2048},
    "nasnet":       {"pool0":44,"pool1": 352, "pool2": 1056, "out2": 1056},
    "densenet_121": {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512, "out3": 1024},
    'vgg11':        {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512,"out4":512},
    'vgg13':        {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512, "pool4": 512,"out4":512},
    'vgg16':        {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512, "pool4": 512,"out4":512},
    'vgg19':        {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512, "pool4": 512,"out4":512},
    'vgg11_bn':     {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512, "pool4": 512,"out4":512},
    'vgg13_bn':     {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512, "pool4": 512,"out4":512},
    'vgg16_bn':     {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512, "pool4": 512,"out4":512},
    'vgg19_bn':     {"pool0":64,"pool1": 128, "pool2": 256, "pool3": 512, "pool4": 512,"out4":512},
    'xception_41':              {"pool1": 128, "pool2": 256, "pool3": 728, "pool4": 2048,"out4":2048},
    'attention_56': {"pool0":64,"pool1": 256, "pool2": 512, "pool3": 1024, "pool4": 2048,"out4":2048},
    'attention_92': {"pool0":64,"pool1": 256, "pool2": 512, "pool3": 1024, "pool4": 2048,"out4":2048}
}
def get_scope_table(encoder_name,poolnum=4,out=False):
    # assert encoder_name in list(scope_table.keys())
    if out==True:
        try:
            return scope_table[encoder_name]['out'+str(poolnum)]
        except:
            raise Exception("Invalid level Encoder! Net do no support this Encoder")
    else:
        try:
            return scope_table[encoder_name]['pool'+str(poolnum)]
        except:
            raise Exception("Invalid level Encoder! Net do no support this Encoder")