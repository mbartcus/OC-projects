from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, MaxPooling2D, Dropout, Reshape, Lambda, UpSampling2D, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.metrics import MeanIoU, OneHotMeanIoU
import tensorflow as tf
import os
import time
import pandas as pd
import json
from keras.models import load_model
import utils

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    if n_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model


def build_vgg16_unet(input_shape, nb_classes):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG16 Model """
    vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = vgg16.get_layer("block1_conv2").output         ## (512 x 512)
    s2 = vgg16.get_layer("block2_conv2").output         ## (256 x 256)
    s3 = vgg16.get_layer("block3_conv3").output         ## (128 x 128)
    s4 = vgg16.get_layer("block4_conv3").output         ## (64 x 64)

    """ Bridge """
    b1 = vgg16.get_layer("block5_conv3").output         ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    if nb_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    """ Output """
    outputs = Conv2D(nb_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="VGG16_U-Net")
    return model


def build_resnet50_unet(input_shape, nb_classes):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.layers[0].output                      ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)                     ## (64 x 64)
    d2 = decoder_block(d1, s3, 256)                     ## (128 x 128)
    d3 = decoder_block(d2, s2, 128)                     ## (256 x 256)
    d4 = decoder_block(d3, s1, 64)                      ## (512 x 512)

    if nb_classes == 1:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    """ Output """
    outputs = Conv2D(nb_classes, 1, padding="same", activation=activation)(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model


def DilatedNet(img_height, img_width, nclasses, use_ctx_module=False, bn=False):
    print('. . . . .Building DilatedNet. . . . .')
    def bilinear_upsample(image_tensor):
        upsampled = tf.image.resize_bilinear(image_tensor, size=(img_height, img_width))
        return upsampled
    
    def conv_block(conv_layers, tensor, nfilters, size=3, name='', padding='same', dilation_rate=1,pool=False):
        if dilation_rate == 1:
            conv_type = 'conv'
        else:
            conv_type = 'dilated_conv'
        for i in range(conv_layers):
            tensor = Conv2D(nfilters, size, padding=padding, use_bias=False, dilation_rate=dilation_rate, name=f'block{name}_{conv_type}{i+1}')(tensor)
            if bn:
                tensor = BatchNormalization(name=f'block{name}_bn{i+1}')(tensor)
            tensor = Activation('relu', name=f'block{name}_relu{i+1}')(tensor)
        if pool:
            tensor = MaxPooling2D(2, name=f'block{name}_pool')(tensor)
        return tensor
       
    nfilters = 64
    img_input = Input(shape=(img_height, img_width, 3))
    x = conv_block(conv_layers=2,tensor=img_input, nfilters=nfilters*1, size=3, pool=True, name=1)
    x = conv_block(conv_layers=2,tensor=x, nfilters=nfilters*2, size=3, pool=True, name=2)
    x = conv_block(conv_layers=3,tensor=x, nfilters=nfilters*4, size=3, pool=True, name=3)
    x = conv_block(conv_layers=3,tensor=x, nfilters=nfilters*8, size=3, name=4)
    x = conv_block(conv_layers=3,tensor=x, nfilters=nfilters*8, size=3,dilation_rate=2, name=5)
    x = conv_block(conv_layers=1,tensor=x, nfilters=nfilters*64, size=7,dilation_rate=4, name='_FCN1')
    x = Dropout(0.5)(x)
    x = conv_block(conv_layers=1,tensor=x, nfilters=nfilters*64, size=1, name='_FCN2')
    x = Dropout(0.5)(x)  
    x = Conv2D(filters=nclasses, kernel_size=1, padding='same', name=f'frontend_output')(x)
    if use_ctx_module:
        x = conv_block(conv_layers=2, tensor=x, nfilters=nclasses*2, size=3, name='_ctx1')
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*4, size=3, name='_ctx2', dilation_rate=2)
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*8, size=3, name='_ctx3', dilation_rate=4)
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*16, size=3, name='_ctx4', dilation_rate=8)
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*32, size=3, name='_ctx5', dilation_rate=16)        
        x = conv_block(conv_layers=1, tensor=x, nfilters=nclasses*32, size=3, name='_ctx7')
        x = Conv2D(filters=nclasses, kernel_size=1, padding='same', name=f'ctx_output')(x)
    x = Lambda(bilinear_upsample, name='bilinear_upsample')(x)
    x = Reshape((img_height*img_width, nclasses))(x)
    x = Activation('softmax', name='final_softmax')(x)
  
    model = Model(inputs=img_input, outputs=x, name='DilatedNet')
    print('. . . . .Building network successful. . . . .')
    return model

def DeeplabV3Plus(image_size, num_classes):
    model_input = Input(shape=(image_size, image_size, 3))
    resnet50 = ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output




        

def train_model(model, model_name, loss_function, data_train, data_val, nbr_epochs = 50, learning_rate = 1e-5, patience = 14, monitor_val = 'val_mean_iou', augmented=False, NB_CLASSES=8):
    
    checkpoint_path = 'saved/checkpoint/'
    models_path = 'saved/models/'
    history_path = 'saved/history/'

    if not os.path.isdir('saved'):
        os.makedirs(checkpoint_path)
        os.makedirs(models_path)
        os.makedirs(history_path)
    
    filepath_check = checkpoint_path + model_name + '/'
    model_path = models_path + model_name
    history_path_file = history_path + model_name + '_history.json'
    
    if os.path.exists(model_path):
        
        if os.path.exists(models_path + 'train_times.json'):
            with open(models_path + 'train_times.json', 'r') as f:
                train_times = json.load(f)
        
        if loss_function == "categorical_crossentropy":
            model = load_model(model_path)
        else:
            model = load_model(model_path, custom_objects={'dice_loss': dice_loss,'total_loss': total_loss}, compile=True)
        print(f'Récupération du modèle {model.name}')

        training_time = train_times[model_name]

        with open(history_path_file, 'r') as f:
            history = json.load(f)
        hist_df = pd.DataFrame.from_dict(history)

    else:
        train_times = {}
        
        start_train = time.time()

        # compile
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function, metrics=[OneHotMeanIoU(num_classes=NB_CLASSES, name='mean_iou')])

        mc = ModelCheckpoint(mode='max', filepath=filepath_check, monitor=monitor_val, save_best_only='True', save_weights_only='True')
        es = EarlyStopping(mode='max', monitor=monitor_val, patience=patience, verbose=1)

        callbacks = [mc, es]

        # fit
        if augmented:
            history = model.fit(data_train_augmented, epochs=nbr_epochs, validation_data=data_val_augmented, callbacks=callbacks)
        else:
            history = model.fit(data_train, epochs=nbr_epochs, validation_data=data_val, callbacks=callbacks)

        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history)

        # save history
        with open(history_path_file, mode='w') as f:
            hist_df.to_json(f)

        # Save the model
        model.load_weights(filepath_check)
        model.save(model_path)

        end_train = time.time()
        training_time = end_train - start_train
        train_times[model_name] = training_time

        with open(models_path + 'train_times.json', 'w') as f:
            json.dump(train_times, f)
  
    return model, hist_df, training_time


def predict_model(model, data_val, model_name, loss_type, training_time, show_pred=False):
    synthese = {}
    
    loss_score, mean_iou_score = model.evaluate(data_val)
    print("Pour le meilleur modèle on obtient :")
    print("mean_iou :", mean_iou_score)
    
    print("loss :", loss_score)

    start_pred = time.time()

    y_pred=model.predict(X_test)
    y_pred_argmax=np.argmax(y_pred, axis=3)

    end_pred = time.time()
    predict_time = end_pred - start_pred

    if show_pred:
        for index in range(10):
            print(index)
            plt.imshow(X_test[index])
            plt.imshow(y_pred_argmax[index], alpha = 0.4)
            plt.show()

    synthese[model_name] = utils.add_model_in_synthese(loss_type, loss_score, mean_iou_score, training_time, predict_time)

    return y_pred_argmax

