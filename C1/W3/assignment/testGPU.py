import tensorflow as tf

# Check if TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Print GPU details
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print("Name:", gpu.name, "Type:", gpu.device_type)

# Create a simple model to check if it runs on the GPU
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

#print("Is the model using the GPU?", tf.test.is_gpu_available())
print("Is the model using the GPU?", tf.config.list_physical_devices('GPU'))

# Check TensorFlow CUDA and cuDNN versions
print("TensorFlow CUDA Version:", tf.sysconfig.get_build_info()["cuda_version"])
print("TensorFlow cuDNN Version:", tf.sysconfig.get_build_info()["cudnn_version"])
