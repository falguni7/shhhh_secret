class Config(object):
    def __init__(self):
        # datasets
        self.dataset = 'IOpsCompetition'
        # model configs
        self.input_channels = 1
        self.kernel_size = 4
        self.stride = 1
        self.final_out_channels = 32
        self.project = 2

        self.dropout = 0.45
        # 16->4 32->6 64->10
        self.features_len = 6
        self.window_size = 32
        self.time_step = 32

        # training configs
        self.num_epoch = 300

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4
        self.weight = 5e-4

        # data parameters
        self.drop_last = False
        self.batch_size = 512
        # trend rate
        self.trend_rate = 1.4
        # negative sample rates
        self.rate = 0.6
         # minimum cut length
        self.cut_rate = 24

        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0015
        # Methods for determining thresholds ("direct","fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
        self.discontinuity = 0.25

    # Distributional uncertainty Augmented Learning hyperparameters
        self.dal_inner_iter = 10
        self.dal_gamma = 50 
        # `dal_gamma_beta` controls gamma scheduling (avoid name clash with optimizer betas)
        self.dal_gamma_beta = 0.01  
        self.dal_rho = 0.1   
        self.dal_strength = 0.0001   
        self.dal_warmup = 0
        # initial scale for embedding bias used in inner-loop (emb_bias = scale * randn_like)
        self.dal_emb_init_scale = 0.0001
        self.dal_alpha = 0.1 


