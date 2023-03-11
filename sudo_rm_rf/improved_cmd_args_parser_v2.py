"""!
@brief Improved Experiment Argument Parser for SudoRmRf

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import argparse


def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(
        description='Experiment Argument Parser')
    # ===============================================
    # Datasets arguments
    parser.add_argument("--train", type=str, nargs='+',
                        help="Training dataset",
                        default=None,
                        choices=['WHAM', 'LIBRI2MIX', 'MUSDB', 'FUSS', 'WHAMR', 'MICROSON'])
    parser.add_argument("--val", type=str, nargs='+',
                        help="Validation dataset",
                        default=None,
                        choices=['WHAM', 'LIBRI2MIX', 'MUSDB', 'FUSS', 'WHAMR', 'MICROSON'])
    parser.add_argument("--test", type=str, nargs='+',
                        help="Test dataset",
                        default=None,
                        choices=['WHAM', 'LIBRI2MIX', 'MUSDB', 'FUSS', 'WHAMR', 'MICROSON'])
    parser.add_argument("--train_val", type=str, nargs='+',
                        help="Validation on the training data",
                        default=None,
                        choices=['WHAM', 'LIBRI2MIX', 'WHAMR', 'MICROSON'])
    parser.add_argument("--n_train", type=int,
                        help="""Reduce the number of training
                                samples to this number.""", default=0)
    parser.add_argument("--n_val", type=int,
                        help="""Reduce the number of evaluation
                                samples to this number.""", default=0)
    parser.add_argument("--n_test", type=int,
                        help="""Reduce the number of test
                                    samples to this number.""", default=0)
    parser.add_argument("--n_train_val", type=int,
                        help="""Reduce the number of evaluation
                             samples on the training set.""", default=0)
    parser.add_argument("--audio_timelength", type=float,
                        help="""The timelength of the audio that you want
                                to load in seconds.""",
                        default=4.)
    parser.add_argument("--min_or_max", type=str,
                        help="""Min or max if this applies to the dataset
                        that you use. Min means that the mixture is going to
                        be cropped at the minimum of all sources and for max
                        is going to be zero-padded""",
                        default='min',
                        choices=['min', 'max'])
    parser.add_argument("--zero_pad_audio", action='store_true',
                        help="""If a specific timelength is required all
                        audio sources and mixtures are going to be zero
                        padded in order to have the required length. If not
                        and a specific timelegth is required then the files
                        with less than required legth are not going to be
                        used.""", default=False)
    parser.add_argument("--normalize_audio", action='store_true',
                        help="""Normalize using mean and standard deviation
                        when reading the files processing each audio file.""",
                        default=False)
    parser.add_argument("--normalize_online", action='store_true',
                        help="""Normalize using mean and standard deviation
                        when doing the online mixing.""",
                        default=False)
    parser.add_argument("--mild_target", action='store_true',
                        help="""use anechoic+0.25*(noise+reverb) as target
                        """,
                        default=False)
    parser.add_argument("--online_mix", action='store_true',
                        help="""Shuffle sources and batch utterances for building
                        new mixtures during the train loop""",
                        default=False)
    # ===============================================
    # Separation task arguments
    parser.add_argument("--n_channels", type=int,
                        help="""The number of mixture channels.""",
                        default=1, choices=[1, 2])
    parser.add_argument("--min_num_sources", type=int,
                        help="""The minimum number of sources in a mixture.""",
                        default=1)
    parser.add_argument("--max_num_sources", type=int,
                        help="""The maximum number of sources in a mixture.""",
                        default=4)
    parser.add_argument("--separation_task", type=str,
                        help="The separation task you would like to perform, "
                             "some of the tasks might not be available for "
                             "specific datasets.",
                        default=None,
                        choices=['enh_noisy',
                                 'enh_reverberant', 'enh_noisyreverberant',
                                 'sep_clean', 'sep_noisy', 'sep_reverberant',
                                 'sep_noisyreverberant'])
    # ===============================================
    # Training params
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch.
                                Warning: Cannot be less than the number of
                                the validation samples""", default=4)
    parser.add_argument("--n_epochs", type=int,
                        help="""The number of epochs that the
                            experiment should run""", default=500)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-3)
    parser.add_argument("--divide_lr_by", type=float,
                        help="""The factor that the learning rate
                            would be divided by""", default=3.)
    parser.add_argument("--patience", type=int,
                        help="""Patience until reducing the learning rate .""",
                        default=5)
    parser.add_argument("--optimizer", type=str,
                        help="""The optimizer that you want to use""",
                        default="adam",
                        choices=['adam', 'radam'])
    parser.add_argument("--clip_grad_norm", type=float,
                        help="""The norm value which all gradients
                            are going to be clipped, 0 means that no
                            grads are going to be clipped""",
                        default=5.)
    parser.add_argument("-fs", type=int,
                        help="""Sampling rate of the audio.""", default=8000)
    # ===============================================
    # CometML experiment configuration arguments
    parser.add_argument("-tags", "--cometml_tags", type=str,
                        nargs="+", help="""A list of tags for the cometml
                            experiment.""",
                        default=[])
    parser.add_argument("--experiment_name", type=str,
                        help="""Name of current experiment""",
                        default=None)
    parser.add_argument("--project_name", type=str,
                        help="""Name of current experiment""",
                        default="yolo_experiment")
    # ===============================================
    # Device params
    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+",
                        help="""A list of Cuda IDs that would be
                            available for running this experiment""",
                        default=['0'],
                        choices=['0', '1', '2', '3'])
    parser.add_argument("--n_jobs", type=int,
                        help="""The number of cpu workers for
                                        loading the data, etc.""", default=4)
    # ===============================================
    # Local experiment logging
    parser.add_argument("-elp", "--experiment_logs_path", type=str,
                        help="""Path for logging experiment's audio.""",
                        default=None)
    parser.add_argument("-mlp", "--metrics_logs_path", type=str,
                        help="""Path for logging metrics.""",
                        default=None)
    parser.add_argument("-clp", "--checkpoints_path", type=str,
                        help="""Path for logging checkpoints.""",
                        default=None)
    parser.add_argument("--save_checkpoint_every", type=int,
                        help="""Number of epochs between each model save.""",
                        default=0)
    # ===============================================
    # Separation model (SuDO-RM-RF) params
    parser.add_argument("--out_channels", type=int,
                        help="The number of channels of the internal "
                             "representation outside the U-Blocks.",
                        default=128)
    parser.add_argument("--in_channels", type=int,
                        help="The number of channels of the internal "
                             "representation inside the U-Blocks.",
                        default=512)
    parser.add_argument("--num_blocks", type=int,
                        help="Number of the successive U-Blocks.",
                        default=16)
    parser.add_argument("--upsampling_depth", type=int,
                        help="Number of successive upsamplings and "
                             "effectively downsampling inside each U-Block. "
                             "The aggregation of all scales is performed by "
                             "addition.",
                        default=5)
    parser.add_argument("--group_size", type=int,
                        help="The number of individual computation groups "
                             "applied if group communication module is used.",
                        default=16)
    parser.add_argument("--enc_kernel_size", type=int,
                        help="The width of the encoder and decoder kernels.",
                        default=21)
    parser.add_argument("--enc_num_basis", type=int,
                        help="Number of the encoded basis representations.",
                        default=512)

    # Attentive sudo parameters
    parser.add_argument("--att_dims", type=int,
                        help="The number of attention depth.",
                        default=256)
    parser.add_argument("--att_n_heads", type=int,
                        help="The number of attention heads.",
                        default=4)
    parser.add_argument("--att_dropout", type=float,
                        help="The dropout rate inside the attention layers.",
                        default=0.1)

    parser.add_argument("--model_type", type=str,
                        help="The type of model you would like to use.",
                        default='relu',
                        choices=['relu', 'softmax', 'groupcomm',
                                 'groupcomm_v2', 'causal',
                                 'attention', 'attention_v2',
                                 'attention_v3', 'sepformer'])

    return parser.parse_args()
