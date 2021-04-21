n = 10000 #n must be a square number
test_input = tf.random.normal([n, 100])
predictions = generator(test_input, training=False) 
print(predictions.shape)
midpoint_distribution = np.zeros_like(predictions[1])
for i in range(len(predictions)):
  max = np.amax(predictions[i])
  maxlocation = np.where(predictions == max)
  x = maxlocation[1][0]
  y = maxlocation[2][0]
  midpoint_distribution[x][y] += 1
  print(i)
plt.imshow(midpoint_distribution[:,:,0])
print(midpoint_distribution)
