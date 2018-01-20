import tensorflow as tf
# Define training and validation datasets with the same structure.
training_dataset = tf.data.Dataset.range(100)
#validation_dataset = tf.data.Dataset.range(50)

# A reinitializable iterator is defined by its structure. We could use the
# `output_types` and `output_shapes` properties of either `training_dataset`
# or `validation_dataset` here, because they are compatible.
iterator = tf.data.Iterator.from_structure(training_dataset.output_types)

next_element = iterator.get_next()


training_init_op = iterator.make_initializer(training_dataset)
#validation_init_op = iterator.make_initializer(validation_dataset)

with tf.Session() as sess:
   # Run 20 epochs in which the training dataset is traversed, followed by the
   # validation dataset.
   for i in range(5):
       # Initialize an iterator over the training dataset.
       # print("#########################  ", i)
       sess.run(training_init_op)
       for _ in range(10):
           nel = sess.run(next_element)
           print("train: ", type(nel), nel)

       # Initialize an iterator over the validation dataset.
       # sess.run(validation_init_op)
       # for _ in range(5):
       #     nel = sess.run(next_element)
       #     print("valid: ", type(nel), nel)