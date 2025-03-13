#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=vulcan-jbhuang
#SBATCH --qos=vulcan-default
#SBATCH --mem=32gb  
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --partition=vulcan-ampere
#SBATCH --output=/vulcanscratch/yiranx/codes/wild-gaussians/jobs/slurm_output-%j.out

set -x

module unload cuda/10.2.89
module add cuda/11.8.0 gcc/7.5.0

export WORK_DIR="/vulcanscratch/yiranx/codes/wild-gaussians/jobs/slurm_${SLURM_JOBID}"

cd /vulcanscratch/yiranx/codes/wild-gaussians
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_noEmbed --save-iters 20000
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/272/650/colmap_0 --output wildgaussians_ckpts/272650_noEmbed --save-iters 20000

# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_noEmbed_phototourism --save-iters 20000 --set 'appearance_enabled=false' --set 'uncertainty_mode=disabled' --set 'iterations=70000' --set 'num_sky_gaussians=50000'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/272/650/colmap_0 --output wildgaussians_ckpts/272650_noEmbed_phototourism --save-iters 20000 --set 'appearance_enabled=false' --set 'uncertainty_mode=disabled' --set 'iterations=150000' --set 'num_sky_gaussians=50000'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/133/292/colmap_0 --output wildgaussians_ckpts/133292_noEmbed_phototourism --save-iters 20000 --set 'appearance_enabled=false' --set 'uncertainty_mode=disabled' --set 'iterations=70000' --set 'num_sky_gaussians=1000'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_noEmbed_phototourism_noUnDistort_resize --save-iters 20000::20000 --set 'appearance_enabled=false' --set 'uncertainty_mode=disabled' --set 'iterations=70000' --set 'num_sky_gaussians=50000' --set 'downscale_loaded_factor=2'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/272/650/colmap_0 --output wildgaussians_ckpts/272650_noEmbed_phototourism_noUnDistort_resize --save-iters 30000::30000 --set 'appearance_enabled=false' --set 'uncertainty_mode=disabled' --set 'iterations=70000' --set 'num_sky_gaussians=50000' --set 'downscale_loaded_factor=2'
# nerfbaselines train --method wild-gaussians --data data/mipnerf360/garden/ --output wildgaussians_ckpts/garden_3dgs_wg_iter70_000 --save-iters 30000::30000 --set "depth_mode='disabled'"  --set "num_sky_gaussians=100" --set 'appearance_enabled=false' --set 'uncertainty_mode=disabled'
# nerfbaselines train --method wild-gaussians --data data/mipnerf360/garden_disturbed/ --output wildgaussians_ckpts/garden_perturb_ApprOcclu_wg_iter70_000 --save-iters 30000::30000 --set "depth_mode='disabled'"  --set "num_sky_gaussians=100" --set 'iterations=70000'
nerfbaselines train --method wild-gaussians --data data/mipnerf360/garden_disturbed/ --output wildgaussians_ckpts/garden_perturb_ApprOccluHard_wg_iter70_000 --save-iters 30000::30000 --set "depth_mode='disabled'"  --set "num_sky_gaussians=100" --set 'iterations=70000'

# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_phototourism_noUnDistort --save-iters 30000::30000 --set 'iterations=70000' --set 'num_sky_gaussians=50000'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/272/650/colmap_0 --output wildgaussians_ckpts/272650_phototourism_noUnDistort --save-iters 30000::30000 --set 'iterations=70000' --set 'num_sky_gaussians=50000'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/272/650/colmap_0 --output wildgaussians_ckpts/272650_phototourism_noUnDistort_iter200_000 --save-iters 30000::30000 --set 'num_sky_gaussians=50000'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_phototourism_noUnDistort_iter200_000 --save-iters 30000::30000 --set 'num_sky_gaussians=50000'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_phototourism_noUnDistort_iter400_000 --save-iters 30000::30000 --set 'iterations=400000'
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/133/292/colmap_0 --output wildgaussians_ckpts/133292_phototourism_noUnDistort_iter200_000 --save-iters 30000::30000 --set 'iterations=200000'


# Depth experiments
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_phototourism_noUnDistort_DepthReg_iter200_000 --save-iters 30000::30000 --set "depth_mode='depth_anything_v2'"

# Uncertainty experiments
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_phototourism_noUnDistort_LargeUncertainty_iter200_000 --save-iters 30000::30000 --set "depth_mode='disabled'" --set "uncertainty_regularizer_weight=5.0"  
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_phototourism_noUnDistort_SmallUncertainty_iter200_000 --save-iters 30000::30000 --set "depth_mode='disabled'" --set "uncertainty_regularizer_weight=0.05"  

# Sky
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_phototourism_noUnDistort_SmallSky_iter200_000 --save-iters 30000::30000 --set "depth_mode='disabled'"  --set "num_sky_gaussians=1000"
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/272/650/colmap_0 --output wildgaussians_ckpts/272650_phototourism_noUnDistort_SmallSky_iter200_000 --save-iters 30000::30000 --set "depth_mode='disabled'"  --set "num_sky_gaussians=1000"
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/124/127/colmap_0 --output wildgaussians_ckpts/124127_phototourism_noUnDistort_100Sky_iter200_000 --save-iters 30000::30000 --set "depth_mode='disabled'"  --set "num_sky_gaussians=100"
# nerfbaselines train --method wild-gaussians --data /vulcanscratch/yiranx/denso_obrm/megascenes/028/023/colmap_0 --output wildgaussians_ckpts/028023_phototourism_noUnDistort_100Sky_iter200_000 --save-iters 30000::30000 --set "depth_mode='disabled'"  --set "num_sky_gaussians=100"

# reproduce
# nerfbaselines train --method wild-gaussians --data external://phototourism/trevi-fountain --output wildgaussians_ckpts/trevi-fountain_phototourism_reproduce_iter200_000 --save-iters 30000::30000
# nerfbaselines train --method wild-gaussians --data external://phototourism/brandenburg-gate --output wildgaussians_ckpts/brandenburg-gate_phototourism_iter200_000 --save-iters 30000::30000