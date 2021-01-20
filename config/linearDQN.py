class Lin_config():
    scaler = 255

    # output config
    output_path  = "results/q2_linear/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"

    # model and training config
    num_episodes_test = 20
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 5000
    log_freq          = 50
    eval_freq         = 1000
    soft_epsilon      = 0

    # hyper params
    nsteps_train       = 2
    nsteps_eval        = 0
    batch_size         = 32
    buffer_size        = 1000
    swap_sch           = 2000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 4
    lr                 = 0.005

    lr_end             = 0.001
    lr_nsteps          = nsteps_train/2
    eps_begin          = 1
    eps_end            = 0.01
    eps_nsteps         = nsteps_train/2
    learning_start     = 200
