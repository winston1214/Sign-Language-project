wget https://raw.github.com/circulosmeos/gdown.pl/master/gdown.pl
chmod u+x gdown.pl
mkdir detector/yolo/data
./gdown.pl 'https://drive.google.com/file/d/1fwl4Hpeuu5fneIGWXsUJK0RBvbMDpISP' detector/yolo/data/yolov3-spp.weights
./gdown.pl 'https://drive.google.com/file/d/1nTJ04JBuETWjj-pAF37tR-JunVEW9H46' pretrained_models/halpe136_fast_res50_256x192.pth
./gdown.pl 'https://drive.google.com/file/d/1ncYuFdP6u31C-KUgUgLLsTMlI2q4Fr5k' best_model.pt