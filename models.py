#input layer
generator = tf.keras.Sequential()
generator.add(Dense(4*4*256,use_bias = False, input_shape = (100,)))
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU(0.2))
generator.add(layers.Reshape((4,4,256)))
assert generator.output_shape == (None, 4, 4, 256)
 
#layer 1
 
generator.add(layers.UpSampling2D((2,2)))
generator.add(layers.Conv2DTranspose(128,
                                (3, 3),
                                strides=(1, 1),
                                padding='same',
                                use_bias = False))
assert generator.output_shape == (None, 8, 8, 128)
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU(0.2))
 
#layer 2
 
generator.add(layers.UpSampling2D((2,2)))
generator.add(layers.Conv2DTranspose(64,
                                (3,3),
                                strides=(1, 1),
                                padding='same',
                                use_bias = False))
assert generator.output_shape == (None,16,16,64)
generator.add(layers.BatchNormalization())
generator.add(layers.LeakyReLU(0.2))
 
#layer 3 / output layer
generator.add(layers.UpSampling2D((2,2)))
generator.add(layers.Conv2DTranspose(1,
                                (3,3),
                                strides=(1, 1),
                                padding='same',
                                use_bias = False,
                                activation='tanh'))   
assert generator.output_shape == (None,32,32 ,1)
generator.add(layers.BatchNormalization())
generator.add(layers.Cropping2D((2,2)))

generator.summary()
generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4, beta_1 = 0.5, beta_2 = 0.9)

############################################################################################################
# layer 1
discriminator = tf.keras.Sequential()
discriminator.add(layers.InputLayer(input_shape = (28, 28, 1)))
discriminator.add(layers.ZeroPadding2D((2,2)))
discriminator.add(layers.Conv2D(64,
                        (5,5),
                        strides=(2,2),
                        padding = 'same',
                        use_bias = True))
discriminator.add(layers.LeakyReLU(0.2))
 
#layer 2
discriminator.add(layers.Conv2D(128,
                        (5,5),
                        strides=(2, 2),
                        padding = 'same',
                        use_bias = True))
discriminator.add(layers.LeakyReLU(0.2))    
discriminator.add(layers.Dropout(0.3)) 

#layer 3
 
discriminator.add(layers.Conv2D(256,
                        (5,5),
                        strides=(2, 2),
                        padding = 'same',
                        use_bias = True))
discriminator.add(layers.LeakyReLU(0.2))    
discriminator.add(layers.Dropout(0.3))

#layer 4
 
discriminator.add(layers.Conv2D(512,
                        (5,5),
                        strides=(2, 2),
                        padding = 'same',
                        use_bias = True,
                        activation = 'linear'))
discriminator.add(layers.LeakyReLU(0.2))   
discriminator.add(layers.Flatten()) 
discriminator.add(layers.Dropout(0.3))    
discriminator.add(Dense(1))
 
discriminator.summary()
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4, beta_1 = 0.5, beta_2 = 0.9)
