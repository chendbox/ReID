# ReID Person Re-identification
ReID pedestrian 

# problem statement

nowadays China spends millions of dollars to create smart city, which based on SkyNet project, the most created video secrity system in most cities. The goal is to build security and safety environments and prompt the stable development of Chinese economy. Person re-identification is not new to data scientist, software engineer, but to create high accuracy and reliable identification algorthims to predict the correct identities must be considered. 

We have cameras around cities, parks, shopping malls, airports and communities. Each person could be captured by several cameras at the same time. However, there are always blank spots since it's impossible to set up cameras everywhere due to the costs. Also the camera angles for capturing images are not always ideal for every pedestrians. The images captured by cameras must be identified and analyzed to have business values. But we often captures no face, backs, lean down, and change fits persons. We can't guarantee to catch every single movement of a person without interruption. If a person walks from home to shopping mall, he would be captured by several cameras and tracking his routes. He could take off his yellow coat in shopping mall and carry new bought backpack and sit on bench to rest. So do the camera recognize the same person as two different ones? or same one. State-of-art methods have been proposed based on deep nerual network. 

In this project, we are targeting a small network MobileNetV2 on small image datasets and trying to build an identification process on pediatricians and validating the models by learning rate decay, label smoothing, batch normalization, data generator, similarity analysis, etc. 


# Contents

| files | description |
| ---| --- |
| Images | Marketing data from JulyEDU on capturing persons images (few) |
| Readme| Read me including details on this project |
| notebook | all-in-one jupyter notebook including details of ReID |
| photos | the ReID reviews in smart city concepts |

* Due to the 25MB file loading size rule, I can't upload the whole images. Feel free to contact me if interested: chendbox@gmail.com

# Workflow

1. set up a baseline for Person ReID (mobilenetV2) on market1501 images files

  training data: '00001': [img1, img2, img3,...]

  testing data: [img ...]



2. Given a query photo, recognize person ID  < == image search / retrieval

   Gallery : image database <== training data + testing data (only a few)



3. q, g_i in G : learn a (similarity / distance) function: s(q, g_i) 

  Suppose q and g_i are feature vectors, s(q, g_i)  = q dot_product g_i 

 

4. future work:  ConvNet e.g. ResNet-50 / Alexnet  == > Train a ConvNet



# Executive Summary

I loaded the images data and found 12936 images in training dataset. Removed duplicates, the number of person id is 751. I used LabelEncoder to label the data.

After train test split, training images is 10348, val images is 2588, and image labels is 751. Implemented a MobileNetV2 backbone and built dense layers, activation function and output layers with Softmax. In the model validation process, I updated the loss functions Softmax output and dense normalization, so there will be two loss functions in the last model. The reason is the application on one-hot encoding and label smoothing. So cross_entropy_label_smoothing and categorical_crossentropy are the two loss functions. 

Step by step, I sampled in triplet data(anchor, positive, negative), anchor shared the same Id with positive and different from negative. In feature space, dis(pos, anchor) + margin < dis(neg, anchor) which I used hinge loss. I created load_img_batch function to load images with X, Y-batch. I created generator_batch_triplet to generate batch data for network loading. I random resampled the feature and label set and did data augmentation on training set. I set up a checkpoint and used to update the learning. Fitted model with 20 epcochs, with a learning rate started at 0.01. After running all epochs, baseline loss from 8.4180 to 6.8080, triplet loss from 0.2886 to 0.2122. The triplet loss was better than baseline loss. Regarding the accuracy, there was no trend which accuracy could be better since this trail run with few epochs due to the computation power. Future work will involve CNN and more data.

I also set up an evaluation process. There is 3368 images in the query dataset, and 25259 images in the gallery datasets. I set up the query image name, with query image path, as well as gallery image and path. Loaded earlier model using cross_entropy_label_smoothing model to get the output layers and extracted features. The passed through the query generator and query features prediction, the query features shape after l2 normalization is 3368 and 1280. Same process with gallery data and features, and gallery feature shape 25259, 1280. I did similarity analysis with matrix dot-product by query feature and gallery features. The prediction is each row in idx_list of the matrix. Then the accuracy of the model is the sum of prediction iterations over lenght of queryimage name. I did not run this but set up a way to help you to evaluate the model. 



# Conclusions

1. I am able to set up a baseline mobilenetV2 with training accuracy 95.6% and testing 84%. 

2. I updated the mobilenetV2 model and getting better performance.

3. Resnet, CNN large network will be used later to extract features.
