from tensorflow.keras.layers import Lambda, Input, Conv2D, ZeroPadding2D, Activation, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.image import resize

def get_discriminator(im_h, im_w, n_class, lrelu_alpha=0.2):

    disc_inp = Input((im_h, im_w, n_class))

    x = ZeroPadding2D(padding=(1,1))(disc_inp)
    x = Conv2D(64, kernel_size=(4,4), strides=2, padding='valid', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(128, kernel_size=(4,4), strides=2, padding='valid', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(256, kernel_size=(4,4), strides=2, padding='valid', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(512, kernel_size=(4,4), strides=2, padding='valid', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)

    preds = Conv2D(1, kernel_size=(4,4), strides=2, padding='valid', activation=None)(x)

    upsample_layer = Lambda(lambda x: resize(x, size=(im_h, im_w), method='bilinear'))

    preds_upsample = upsample_layer(preds)
    preds_upsample = Activation('sigmoid')(preds_upsample)

    return Model(disc_inp, preds_upsample, name='discriminator')

def get_discriminator_tiny(im_h, im_w, n_class, lrelu_alpha=0.2):

    disc_inp = Input((im_h, im_w, n_class))

    x = ZeroPadding2D(padding=(1,1))(disc_inp)
    x = Conv2D(128, kernel_size=(4,4), strides=2, padding='valid', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(256, kernel_size=(4,4), strides=2, padding='valid', activation=None)(x)
    x = LeakyReLU(alpha=0.2)(x)

    preds = Conv2D(1, kernel_size=(4,4), strides=2, padding='valid', activation=None)(x)

    upsample_layer = Lambda(lambda x: resize(x, size=(im_h, im_w), method='bilinear'))

    preds_upsample = upsample_layer(preds)
    preds_upsample = Activation('sigmoid')(preds_upsample)

    return Model(disc_inp, preds_upsample, name='discriminator')
