USER=gkyela123
IMAGE=halkspeach
VERSION=1.0.1

docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t docker.io/$USER/$IMAGE:$VERSION \
  -t docker.io/$USER/$IMAGE:latest \
  --push .


  docker run -d --name halkspeach --gpus all -p 8000:8000 docker.io/gkyela123/halkspeach:1.0.2




  Willy030125/whisper_large_v3_turbo_noise_redux_v5_ct2

curl -s "localhost:8000/v1/audio/transcriptions" -F "file=@gkytest.wav" -F "model=deepdml/faster-whisper-large-v3-turbo-ct2"
curl "localhost:8000/v1/models/deepdml/faster-whisper-large-v3-turbo-ct2" -X POST
