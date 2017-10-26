####################################
#data parameters
image_size = 32*32
image_shape = (32, 32)
num_channel = 3
####################################
#network parameters
nc_netG = 3
ndf_netG = 16
#ngf_netG = 256
#nz_netG =128
#ninput_netG = 8*ndf_netG + nz_netG


nc_netD = 3
ndf_netD = 256

####################################
#training parameters
epoch_num = 40
lr_D = 0.001
lr_G = 0.0005


batch_size = 64
test_batch_size = 1000
###################################
###perturbation magnitude for training
coef_FGSM = 0.03
coef_L2 = 1.0

coef_FGSM_gap = 0.3
coef_L2_gap = 3.0

coef_FGSM_ll = 0.3
coef_L2_ll = 3.0

coef_gan = 0.47