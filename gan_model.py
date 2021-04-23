from tensorflow import keras
import tensorflow as tf
from model import Deeplabv3
from model_discriminator import get_discriminator, get_discriminator_tiny
import os

class AdverGAN(keras.Model):
  def __init__(self, lambda_adv, target_size, n_class, gen_model_path=None, disc_model_path=None):
    super(AdverGAN, self).__init__()
    if gen_model_path is not None:
      self.gen = load_model(gen_model_path)
    else:
      self.gen = Deeplabv3(input_shape=(target_size[0], target_size[1], 3), backbone='xception', activation='softmax')
    if disc_model_path is not None:
      self.disc = load_model(disc_model_path)
    else:
      self.disc = get_discriminator_tiny(target_size[0], target_size[1], n_class)
    self.lambda_adv = lambda_adv
    self.t_size = target_size
  
  def compile(self, g_opt, d_opt, gan_loss, gan_metric, disc_metric_fn,
              g_loss_fn, g_metric_fn, g_tracker, gan_tracker, disc_tracker):
    
    super(AdverGAN, self).compile()
    self.gen_opt = g_opt
    self.disc_opt = d_opt
    self.gan_loss_fn = gan_loss
    self.gan_metric_fn = gan_metric
    self.g_loss_fn = g_loss_fn
    self.g_metric_fn = g_metric_fn
    self.disc_metric_fn = disc_metric_fn
    self.g_metric = g_tracker
    self.gan_metric = gan_tracker
    self.disc_metric = disc_tracker

  def call(self, x):
    pred_imgs = self.gen(x)
    return self.disc(pred_imgs)

  @property
  def metrics(self):
    return [self.g_metric, self.gan_metric, self.disc_metric]

  def train_step(self, data):
    imgs, masks = data
    # Generate images
    generated_masks = self.gen(imgs)

    real_img_labels = tf.ones((tf.shape(imgs)[0], self.t_size[0], self.t_size[1], 1))
    fake_img_labels = tf.zeros((tf.shape(imgs)[0], self.t_size[0], self.t_size[1], 1))

    # Train discriminator with real images
    with tf.GradientTape() as tape:
      disc_real_preds = self.disc(masks, training=True)
      disc_real_loss = self.gan_loss_fn(real_img_labels, disc_real_preds)
    
    disc_real_grads = tape.gradient(disc_real_loss, self.disc.trainable_variables)
    self.disc_opt.apply_gradients(zip(disc_real_grads, self.disc.trainable_variables))

    # Train discriminator with fake images
    with tf.GradientTape() as tape:
      disc_fake_preds = self.disc(generated_masks, training=True)
      disc_fake_loss = self.gan_loss_fn(fake_img_labels, disc_fake_preds)
    
    disc_fake_grads = tape.gradient(disc_fake_loss, self.disc.trainable_variables)
    self.disc_opt.apply_gradients(zip(disc_fake_grads, self.disc.trainable_variables))

    # Train generator
    with tf.GradientTape() as tape:
      pred_masks = self.gen(imgs, training=True)
      g_loss = tf.reduce_mean(self.g_loss_fn(masks, pred_masks))
      disc_preds_on_fake = self.disc(pred_masks)
      gan_loss = tf.reduce_mean(self.gan_loss_fn(real_img_labels, disc_preds_on_fake))
      total_g_loss = g_loss + self.lambda_adv * gan_loss
    
    gen_grads = tape.gradient(total_g_loss, self.gen.trainable_variables)
    self.gen_opt.apply_gradients(zip(gen_grads, self.gen.trainable_variables))

    self.g_metric.update_state(self.g_metric_fn(masks, pred_masks))
    self.gan_metric.update_state(self.gan_metric_fn(real_img_labels, disc_preds_on_fake))
    disc_fake_acc = self.disc_metric_fn(fake_img_labels, disc_fake_preds)
    disc_real_acc = self.disc_metric_fn(real_img_labels, disc_real_preds)

    self.disc_metric.update_state( (disc_fake_acc + disc_real_acc) / 2 )

    return {"disc_fake_loss": disc_fake_loss, "disc_real_loss": disc_real_loss,
            "gan_loss": total_g_loss, "gen_iou": self.g_metric.result(), "g_loss": g_loss,
            "disc_acc": self.disc_metric.result(), "gan_acc": self.gan_metric.result()}


  def test_step(self, data):
    imgs, masks = data

    generated_masks = self.gen(imgs, training=False)

    real_img_labels = tf.ones((tf.shape(imgs)[0], self.t_size[0], self.t_size[1], 1))
    fake_img_labels = tf.zeros((tf.shape(imgs)[0], self.t_size[0], self.t_size[1], 1))

    # DISC
    disc_fake_preds = self.disc(generated_masks, training=False)
    disc_real_preds = self.disc(masks, training=False)
    
    disc_fake_loss = self.gan_loss_fn(fake_img_labels, disc_fake_preds)
    disc_real_loss = self.gan_loss_fn(real_img_labels, disc_real_preds)

    disc_fake_acc = self.disc_metric_fn(fake_img_labels, disc_fake_preds)
    disc_real_acc = self.disc_metric_fn(real_img_labels, disc_real_preds)
    self.disc_metric.update_state( (disc_fake_acc + disc_real_acc) / 2 )

    # GEN
    gen_iou = self.g_metric_fn(masks, generated_masks)
    self.g_metric.update_state(gen_iou)
    g_loss = tf.reduce_mean(self.g_loss_fn(masks, generated_masks))
    # GAN
    gan_loss = tf.reduce_mean(self.gan_loss_fn(real_img_labels, disc_fake_preds))
    gan_acc = self.gan_metric_fn(real_img_labels, disc_fake_preds)
    self.gan_metric.update_state(gan_acc)
    
    total_g_loss = g_loss + self.lambda_adv * gan_loss

    return {"disc_fake_loss": disc_fake_loss, "disc_real_loss": disc_real_loss,
        "gan_loss": total_g_loss, "gen_iou": self.g_metric.result(), "g_loss": g_loss,
        "disc_acc": self.disc_metric.result(), "gan_acc": self.gan_metric.result()}

  def save_models(self, models_save_dir):
    self.gen.save(os.path.join(models_save_dir, "generator_tiny.h5"))
    self.disc.save(os.path.join(models_save_dir, "discriminator_tiny.h5"))
