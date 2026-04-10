# 1/ Train Garden
# # Bước 1: Train
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=garden python train.py -s ./datasets/mipnerf360/garden -i images_original --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008 

# # Bước 2: Render (Tạo ảnh kết quả từ model vừa train)
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/garden --skip_train

# # Bước 3: Metrics (Chấm điểm PSNR/SSIM)
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/garden

# #2/ Train Bicycle
# # Bước 1: Train
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bicycle python train.py -s ./datasets/mipnerf360/bicycle -i images_original --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008

# # Bước 2: Render (Tạo ảnh kết quả từ model vừa train)
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/bicycle --skip_train

# # Bước 3: Metrics (Chấm điểm PSNR/SSIM)
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/bicycle

# #3/ Train bonsai
# # Bước 1: Train
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=bonsai python train.py -s ./datasets/mipnerf360/bonsai -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008

# # Bước 2: Render (Tạo ảnh kết quả từ model vừa train)
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/bonsai --skip_train

# # Bước 3: Metrics (Chấm điểm PSNR/SSIM)
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/bonsai

# #4/ Train counter
# # Bước 1: Train
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=counter python train.py -s ./datasets/mipnerf360/counter -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008

# # Bước 2: Render (Tạo ảnh kết quả từ model vừa train)
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/counter --skip_train

# # Bước 3: Metrics (Chấm điểm PSNR/SSIM)
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/counter

# #5/ Train flowers
# # Bước 1: Train
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=flowers python train.py -s ./datasets/mipnerf360/flowers -i images --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008

# # Bước 2: Render (Tạo ảnh kết quả từ model vừa train)
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/flowers --skip_train

# # Bước 3: Metrics (Chấm điểm PSNR/SSIM)
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/flowers

# #6/ Train kitchen
# # Bước 1: Train
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen python train.py -s ./datasets/mipnerf360/kitchen -i images_2 --eval --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008

# # Bước 2: Render (Tạo ảnh kết quả từ model vừa train)
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/kitchen --skip_train

# # Bước 3: Metrics (Chấm điểm PSNR/SSIM)
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/kitchen

# #6/ Train kitchen (add Dash)
# # Bước 1: Train
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen python train.py -s ./datasets/mipnerf360/kitchen -i images_8 --eval --dash --densification_interval 500  --optimizer_type default --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0008

# # Bước 2: Render (Tạo ảnh kết quả từ model vừa train)
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/kitchen --skip_train

# # Bước 3: Metrics (Chấm điểm PSNR/SSIM)
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/kitchen

#6/ Train kitchen (add Dash + sparse_adam + VCD update)
# Bước 1: Train
CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen python train.py -s ./datasets/mipnerf360/kitchen -i images_2 --eval  --dash --dash_warmup_iters 1500 --preset_upperbound 2500000 --optimizer_type default  --densification_interval 500 --test_iterations 30000 --highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0008

# Bước 2: Render (Tạo ảnh kết quả từ model vừa train)
CUDA_VISIBLE_DEVICES=0 python render.py -m output/kitchen --skip_train

# Bước 3: Metrics (Chấm điểm PSNR/SSIM)
CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/kitchen

# # Train_big kitchen
# CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID=kitchen_big python train.py -s ./datasets/mipnerf360/kitchen -i images_2 --eval --dash --dash_warmup_iters 1500 --preset_upperbound 1500000 --densification_interval 100  --optimizer_type default --test_iterations 30000  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0003
# CUDA_VISIBLE_DEVICES=0 python render.py -m output/kitchen_big --skip_train
# CUDA_VISIBLE_DEVICES=0 python metrics.py -m output/kitchen_big