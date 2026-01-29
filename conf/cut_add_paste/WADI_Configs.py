class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'WADI'
        # model configs
        self.input_channels = 127
        self.kernel_size = 4
        self.stride = 1
        self.final_out_channels = 32
        self.project = 2

        self.dropout = 0.45
        # 8->3 16->4 32->6 64->10, 128->
        self.features_len = 8   
        self.window_size = 48   
        self.time_step = 16

        # training configs
        self.num_epoch = 50

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight = 5e-3

        # data parameters
        self.drop_last = False
        self.batch_size = 512
        # trend rate
        self.trend_rate = 0.1
        self.rate = 1
        # number of trend dimensions
        self.dim = 15
        # minimum cut length
        self.cut_rate = 16

        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.001
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
        self.discontinuity = 0.25

    # Distributional uncertainty Augmented Learning hyperparameters
        self.dal_inner_iter = 10    
        self.dal_gamma = 50 
        # `dal_gamma_beta` controls gamma scheduling (avoid name clash with optimizer betas)
        self.dal_gamma_beta = 0.001  
        self.dal_rho = 50   
        self.dal_strength = 0.001 
        self.dal_warmup = 0
        # initial scale for embedding bias used in inner-loop (emb_bias = scale * randn_like)
        self.dal_emb_init_scale = 0.0001
        self.dal_alpha = 1  



