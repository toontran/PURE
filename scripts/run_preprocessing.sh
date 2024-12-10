python3 -m experiments.preprocessing.train_detection --config_file "configs/imagenet/resnet34_timm.yaml" || { echo "failed: python3 -m experiments.preprocessing.train_detection --config_file \"configs/imagenet/resnet34_timm.yaml\"" ; exit 1; }
# python3 -m experiments.preprocessing.train --config_file "configs/imagenet/resnet34_timm.yaml" || { echo "failed: python3 -m experiments.preprocessing.train --config_file \"configs/imagenet/resnet34_timm.yaml\"" ; exit 1; }

python3 -m experiments.preprocessing.get_loss --config_file "configs/imagenet/resnet34_timm.yaml" || { echo "failed: python3 -m experiments.preprocessing.get_loss --config_file \"configs/imagenet/resnet34_timm.yaml\"" ; exit 1; }


# python3 -m experiments.preprocessing.crp_run --config_file "configs/imagenet/resnet50_timm.yaml"
# python3 -m experiments.preprocessing.crp_run --config_file "configs/imagenet/resnet34_timm.yaml" --split train || { echo "failed: python3 -m experiments.preprocessing.crp_run --config_file \"configs/imagenet/resnet34_timm.yaml\" --split train " ; exit 1; }
# python3 -m experiments.preprocessing.crp_run --config_file "configs/imagenet/resnet34_timm.yaml" --split val || { echo "failed: python3 -m experiments.preprocessing.crp_run --config_file \"configs/imagenet/resnet34_timm.yaml\" --split val" ; exit 1; }
#python3 -m experiments.preprocessing.crp_run --config_file "configs/imagenet/resnet101_timm.yaml"

# python3 -m experiments.preprocessing.compute_latent_features --config_file "configs/imagenet/resnet50_timm.yaml"
# python3 -m experiments.preprocessing.compute_latent_features --config_file "configs/imagenet/resnet34_timm.yaml" --split train || { echo "failed: python3 -m experiments.preprocessing.compute_latent_features --config_file \"configs/imagenet/resnet34_timm.yaml\" --split train" ; exit 1; }
# python3 -m experiments.preprocessing.compute_latent_features --config_file "configs/imagenet/resnet34_timm.yaml" --split val || { echo "failed: python3 -m experiments.preprocessing.compute_latent_features --config_file \"configs/imagenet/resnet34_timm.yaml\" --split val" ; exit 1; }
#python3 -m experiments.preprocessing.compute_latent_features --config_file "configs/imagenet/resnet101_timm.yaml"

# python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenet/resnet50_timm.yaml"
# python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenet/resnet34_timm.yaml" --split train || { echo "failed: python3 -m experiments.preprocessing.compute_embeddings --config_file \"configs/imagenet/resnet34_timm.yaml\" --split train" ; exit 1; }
# python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenet/resnet34_timm.yaml" --split val || { echo "failed: python3 -m experiments.preprocessing.compute_embeddings --config_file \"configs/imagenet/resnet34_timm.yaml\" --split val" ; exit 1; }
#python3 -m experiments.preprocessing.compute_embeddings --config_file "configs/imagenet/resnet101_timm.yaml"



python3 -m experiments.preprocessing.crp_run_detection --config_file "configs/imagenet/resnet34_timm.yaml" --split train || { echo "failed: python3 -m experiments.preprocessing.crp_run_detection --config_file \"configs/imagenet/resnet34_timm.yaml\" --split train " ; exit 1; }
python3 -m experiments.preprocessing.compute_latent_features_detection --config_file "configs/imagenet/resnet34_timm.yaml" --split train || { echo "failed: python3 -m experiments.preprocessing.compute_latent_features --config_file \"configs/imagenet/resnet34_timm.yaml\" --split train" ; exit 1; }
