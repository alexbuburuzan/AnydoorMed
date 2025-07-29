# AnydoorMed: Reference-Guided Anomaly Inpainting for Medical Counterfactuals

wget https://huggingface.co/spaces/xichenhku/AnyDoor/resolve/main/epoch=1-step=8687.ckpt
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
python extract_autoencoder.py 


take the best checkpoints (last one in my case) and move and rename it to checkpoints/autoencoder/vae_mammo_0.13rec.ckpt

