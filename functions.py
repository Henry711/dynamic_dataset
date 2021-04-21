def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    total_loss = fake_loss - real_loss
    d = total_loss.numpy()
    daccuracies.append(d)
    return total_loss

def generator_loss(fake_output):
    x = -tf.reduce_mean(fake_output)
    t = x.numpy()
    for i in range(5):
      gaccuracies.append(t)
    return x

def gradient_penalty(real_images, fake_images):

    alpha = tf.random.uniform([BATCH_SIZE,1,1,1],0.0,1.0)
    x = alpha*real_images + ((1-alpha)*fake_images)

    with tf.GradientTape() as tape:
      tape.watch(x)
      pred = discriminator(x)
    grads = tape.gradient(pred, x)[0]
    gradsquared = (grads)**2
    gp1 = tf.reduce_mean(gradsquared)
    gp = 10*gp1
    return gp

def generate_and_save_images(test_input):
  predictions = generator(test_input, training=False)
  predictions = tf.reshape(predictions, (5,28,28))
  fig = plt.figure(figsize=(10,50))
 
  for i in range((predictions.shape[0])):
    plt.subplot(1, 5, i+1)
    plt.imshow(predictions[i,:,:] * 255)
  plt.savefig('image_at_iteration_{:04d}.png'.format(i))
  plt.show()

def train_step(image,batch_size):
  noise = tf.random.normal([batch_size,100])    
  with tf.GradientTape() as gen_tape:
    generated_images = generator(noise)
    fake_output = discriminator(generated_images)
    gen_loss = generator_loss(fake_output)  
  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
 
  for i in range(5):
      with tf.GradientTape() as disc_tape:
        generated_images = generator(noise)
        real_output = discriminator(image)
        fake_output = discriminator(generated_images)
        gp = gradient_penalty(image, generated_images)
        disc_loss = discriminator_loss(real_output, fake_output) 
        d_cost = disc_loss+gp
      gradients_of_discriminator = disc_tape.gradient(d_cost, discriminator.trainable_variables)
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
