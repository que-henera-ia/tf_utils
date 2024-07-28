import time
import tensorflow as tf
from tf_utils.manage_data import *



def train_model(model, train_dataset, epochs, test_dataset=None, num_test_generation=4 ,validation_dataset=None, generation_seed=None, start_epoch=0, results_path="data_out/"):
    '''
    test_dataset: comes from the same dataset than train_dataset. Is used to know the training process on supervised learning 
                    with test_dataset. test_dataset is usually 20% train dataset. GANs usually does not have test_dataset.
    validation_dataset: this dataset contains few samples. These samples must come from other origin than train and test datasets.
                        Is used to really know how the model generalizes.
    generation_seed: it is a random seed of shape [# images to generate, latent dimension]. It is used to generate images with those
                    random seeds. Setting the random seed will make the training visualization better.
    '''  
    test_samples = None
    if test_dataset is not None:
        # Pick a sample of the test set for generating output images
        for test_batch in test_dataset.take(1):
            test_samples = test_batch[0:num_test_generation, :, :, :]

    for epoch in range(start_epoch+1, epochs+start_epoch+1):
        # Train model
        ## TODO: compute train error as test error
        start = time.time()
        for image_batch in train_dataset:
            model.train_step(image_batch)
        print('-- EPOCH {}'.format(epoch))
        print('Execution Time {} sec'.format(time.time()-start))

        show_training_performance(model, epoch, test_dataset=test_dataset, test_samples=test_samples, validation_dataset=validation_dataset, generation_seed=generation_seed, results_path=results_path)

        # Save the model every 15 epochs
        if (epoch) % 15 == 0:
            model.save_weights("{epoch:04d}".format(epoch=epoch))

    # Save model after final epoch
    model.save_weights("{epoch:04d}".format(epoch=epochs))
    
    # Generate after the final epoch
    show_training_performance(model, epoch, test_dataset=test_dataset, test_samples=test_samples, validation_dataset=validation_dataset, generation_seed=generation_seed, results_path=results_path)

    # Generate GIF and video to show training progress
    '''
    gifs not generating properly
    '''
    save_gif(results_path+model.model_name, re_images_name='{}{}-image_at_epoch_*.png'.format(results_path, model.model_name))
    save_mp4(results_path+model.model_name, re_images_name='{}{}-image_at_epoch_*.png'.format(results_path, model.model_name))


def show_training_performance(model, epoch, test_dataset=None, test_samples=None, validation_dataset=None, generation_seed=None, results_path="data_out/"):
    if (test_dataset is None) and (validation_dataset is None) and (generation_seed is None):
        print("WARNING! CANNOT MEASURE PERFORMANCE! No validation dataset, test dataset, or generation seed provided...")

    # Compute test error to know training performance
    if test_dataset is not None:
        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(model.compute_loss(test_x))
        mean_test_loss = -loss.result()

        print('Mean Test Loss:')
        for l, n in zip(mean_test_loss, model.loss_names):
            print('{} : {}'.format(n,l))

        # Produce images for the GIF as you go
        save_image_matrix(model.inference_batch_images(test_samples), img_path ='{}{}-test-image_at_epoch_{:04d}'.format(results_path, model.model_name, epoch))

    # Test training with a validation dataset
    if validation_dataset is not None:
        save_image_matrix(model.inference_batch_images(validation_dataset), img_path ='{}{}-val-image_at_epoch_{:04d}'.format(results_path, model.model_name, epoch))

    # Generate images given random seed
    if generation_seed is not None:
        '''
        TODO: fix the argument generation_seed
        generation_seed must be model.seed
        '''
        save_image_matrix(model.generate_images(model.seed), img_path ='{}{}-generated-image_at_epoch_{:04d}'.format(results_path, model.model_name, epoch))