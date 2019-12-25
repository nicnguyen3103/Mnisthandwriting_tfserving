gsutil cp ./my_custom_code-0.1.tar.gz gs://$BUCKET_NAME/custom_prediction_routine_tutorial/my_custom_code-0.1.tar.gz    
gsutil cp my_model preprocessor.pkl gs://$BUCKET_NAME/custom_prediction_routine_tutorial/model/


MODEL_NAME='mnisthandwriting'
VERSION_NAME='v1'

gcloud components install beta

gcloud beta ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --runtime-version 1.13 \
  --python-version 3.5 \
  --origin gs://$BUCKET_NAME/custom_prediction_routine_tutorial/model/ \
  --package-uris gs://$BUCKET_NAME/custom_prediction_routine_tutorial/my_custom_code-0.1.tar.gz \
  --prediction-class predictor.MyPredictor

# host_path:container_path
docker run -it --rm -p 8500:8500 -p 8501:8501 \
             -v "/home/coderschool/deploying/mnist_fansipan/model_train/my_mnist_model:/models/my_mnist_model" \
             -e MODEL_NAME=my_mnist_model \
             tensorflow/serving