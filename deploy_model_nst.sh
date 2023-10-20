mkdir model_store/nst

torch-model-archiver --model-name nst \
                     --version 1.0 \
                     --serialized-file ./nst.pt \
                     --export-path model_store/nst \
                     --handler handler_nst.py \


mv model_store/nst/nst.mar model_store/nst/model.mar


#rm -r logs
#torchserve --start --model-store model_store/nst --models nst=model.mar


gsutil cp model_store/nst/model.mar gs://tshirt_model/nst/
gsutil cp handler_nst.py gs://tshirt_model/nst/
gsutil cp /home/maktukma/projects/gcp/pytorch-AdaIN/input/style gs://tshirt_model/nst/style

gcloud ai models upload \
  --region=us-central1 \
  --display-name=nst \
  --container-image-uri=us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.1-13:latest \
  --artifact-uri=gs://tshirt_model/nst


gcloud ai endpoints list --region=us-central1
gcloud ai models list --region=us-central1


gcloud ai endpoints deploy-model 2615538051362848768\
  --region=us-central1 \
  --model=6919717055780356096 \
  --display-name=nst \
  --min-replica-count=1 \
  --max-replica-count=1 \
  --traffic-split=0=100


endpoints = aiplatform.Endpoint.list()
for i in endpoints:
        i.undeploy_all()