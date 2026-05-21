# Standard library imports
import argparse
import time
import torch

# Local imports
import replace
from utils import load_dataset, load_sop_dataset, load_model_and_processor, run_training, load_wyze_train_dataset
from utils import load_ym_dataset, setup_logging
from utils.data_utils import load_face_dataset, load_pet_dataset
from utils.data_utils import load_buildings_dataset, load_vehicle_dataset
from utils.data_utils import load_unified_dataset

class TrainingConfig:
    """Configuration class for training parameters."""

    def __init__(self, args):
        self.seed = args.seed
        self.gallery_size = args.gallery_size
        self.learning_rate = args.learning_rate
        self.filter = args.filter
        self.batch_size = args.batch_size
        self.object_type = args.object_type
        self.input_mode = args.input_mode
        self.feature_mode = args.feature_mode
        self.training_parameters = args.training_parameters
        self.test = args.test
        self.captions = args.captions
        self.model_name_or_path = args.model_name_or_path
        self.train_file = args.train_file
        self.data_folder = args.data_folder
        self.prefix = args.prefix
        self.expert_feature = "None" if args.input_mode == "image_only" else args.expert_feature
        self.warmup_connector_path = args.warmup_connector_path
        self.cross_attn_heads = args.cross_attn_heads
        self.qformer_num_queries = args.qformer_num_queries
        self.qformer_num_heads = args.qformer_num_heads
        self.num_train_epochs = args.num_train_epochs

        # Derived configurations
        self.model_id = args.model_name_or_path
        self.model_short_name = self.model_id.split('/')[-1]  # e.g. Qwen2.5-VL-7B-Instruct
        self.run_name = self._generate_run_name()
        print(f"Run name: {self.run_name}")

    def _generate_run_name(self):
        """Generate a unique run name based on configuration."""
        current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        name = (f'runs/{current_time}_{self.prefix}_{self.model_short_name}_{self.object_type}_'
                f'{self.feature_mode}_{self.expert_feature}_{self.input_mode}_'
                f'lr_{self.learning_rate}_bs_{self.batch_size}_captions_{self.captions}')
        return name



def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Training configuration for VLMID model.")
    parser.add_argument("--test", action="store_true", help="Run in test mode (evaluation only).")

    # Model and training configuration
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--gallery_size", type=int, default=5, help="Gallery size for ReID")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--filter", type=float, default=0.5, help="Filter threshold for dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--object_type", type=str, default='person', help="Object type for ReID")
    parser.add_argument("--captions", type=lambda x: x.lower() == 'true', default=False, help="Use captions")
    parser.add_argument("--train_file", type=str, default=None, help="Path to the training dataset file")
    parser.add_argument("--data_folder", type=str, default='', help="Root folder for image paths in the dataset")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="HuggingFace model ID or local path to the model")
    parser.add_argument("--prefix", type=str, default='', help="Prefix appended to the run name")

    # Main combinations (recommended):
    #   1. input_mode='image_only' with feature_mode='vanilla'
    #   2. input_mode='expert_and_image_attn' with feature_mode='expert'
    # Other combinations are available but less commonly used
    parser.add_argument("--feature_mode", type=str, default='vanilla',
                       choices=['vanilla', 'expert', 'random', 'fully_random'],
                       help="Feature mode for expert features. Main combinations: 'vanilla' (with image_only) or 'expert' (with expert_and_image_attn)")

    parser.add_argument("--input_mode", type=str, default='image_only',
                       choices=['expert_and_image_attn', 'expert_and_image_concat', 'expert_and_image_add', 'expert_only', 'image_only', 'expert_cross_attn', 'expert_qformer'],
                       help="Input mode for the model. Main combinations: 'image_only' (with vanilla), 'expert_and_image_attn' (with expert), 'expert_cross_attn' (with expert; per-sample learnable cross-attention), or 'expert_qformer' (with expert; BLIP-2-style learnable-query two-stage fuser)")

    parser.add_argument("--expert_feature", type=str, default="None",
                       choices=["PLIP", "wyzev0323token", "wyzev0202reid", "wyzev0415token", "DINOv2", "None"],
                       help="Expert feature model to use when feature_mode='expert'")

    parser.add_argument("--training_parameters", nargs='+',
                       # default=['merger', 'expert_projector', 'expert'],
                       # ----------Important ablation study---------
                       default=['merger', 'expert_projector'],
                       choices=['llm', 'merger', 'expert_projector', 'expert', 'expert_fuser'],
                       help="Training parameters to update")

    parser.add_argument("--warmup_connector_path", type=str, default=None,
                       help="Optional path to a previously-saved connector.pt whose "
                            "expert_projector weights should be loaded as a warm-start. "
                            "Only keys containing 'expert_projector' are loaded; "
                            "everything else (e.g., a freshly-initialised expert_fuser) "
                            "keeps its init.")

    parser.add_argument("--cross_attn_heads", type=int, default=8,
                       help="Number of attention heads for input_mode='expert_cross_attn'.")

    parser.add_argument("--qformer_num_queries", type=int, default=8,
                       help="Number of learnable query tokens for input_mode='expert_qformer'.")
    parser.add_argument("--qformer_num_heads", type=int, default=8,
                       help="Number of attention heads for input_mode='expert_qformer'.")

    parser.add_argument("--num_train_epochs", type=int, default=1,
                       help="Number of training epochs. A connector checkpoint is "
                            "saved at the end of every epoch under "
                            "<run_name>/ckpts/connector_epoch_<N>.pt, and the "
                            "lowest-eval_loss checkpoint is saved to "
                            "<run_name>/ckpts/connector_best_step_<N>.pt; the final "
                            "<run_name>/connector.pt mirrors the last epoch's checkpoint.")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Create configuration object
    config = TrainingConfig(args)

    # Tee stdout/stderr to runs/<run_name>/train.log
    import os
    setup_logging(os.path.join(config.run_name, 'train.log'))

    # Main combinations (recommended):
    #   1. input_mode='image_only' with feature_mode='vanilla'
    #   2. input_mode='expert_and_image_attn' with feature_mode='expert'
    # Other combinations are available but less commonly used
    main_combinations = [
        ('image_only', 'vanilla'),
        ('expert_and_image_attn', 'expert'),
        ('expert_cross_attn', 'expert'),
        ('expert_qformer', 'expert'),
    ]
    if (config.input_mode, config.feature_mode) not in main_combinations:
        print(f"Note: Using non-standard combination: input_mode='{config.input_mode}' with feature_mode='{config.feature_mode}'")
        print("Main combinations are: ('image_only', 'vanilla'), ('expert_and_image_attn', 'expert'), ('expert_cross_attn', 'expert'), or ('expert_qformer', 'expert')")

    # Load data
    def load_data_by_object_type(cfg):
        loaders = {
            # 'person': lambda: load_dataset(cfg.gallery_size, cfg.filter, cfg.object_type, cfg.captions),
            'person': lambda:load_wyze_train_dataset(cfg, cfg.object_type, cfg.captions),

            'pet': lambda: load_pet_dataset(cfg.gallery_size, cfg.filter, cfg.object_type, cfg.captions),
            'face': lambda: load_face_dataset(cfg.gallery_size, cfg.filter, cfg.object_type, cfg.captions),
            'building': lambda: load_buildings_dataset(cfg.gallery_size, cfg.filter, cfg.object_type, cfg.captions),
            'vehicle': lambda: load_vehicle_dataset(cfg.gallery_size, cfg.filter, cfg.object_type, cfg.captions),
            'sop': lambda: load_sop_dataset(cfg.gallery_size, cfg.filter, cfg.object_type, cfg.captions),
            'unified': lambda: load_unified_dataset(),
        }
        return loaders.get(cfg.object_type, loaders['unified'])()

    train_dataset, eval_dataset = load_data_by_object_type(config)
    # Load model and processor
    model, processor = load_model_and_processor(config)

    # Optional warm-start: load expert_projector weights from a previous connector.pt.
    # Useful when switching fusion strategy (e.g. expert_and_image_attn → expert_cross_attn)
    # so the expert_projector output is already aligned to Qwen's hidden space, and only
    # the new fuser has to learn from scratch.
    if config.warmup_connector_path:
        print(f"[warmup] Loading expert_projector weights from: {config.warmup_connector_path}")
        ckpt = torch.load(config.warmup_connector_path, map_location='cuda')
        filtered = {k: v for k, v in ckpt.items() if 'expert_projector' in k}
        if not filtered:
            print(f"[warmup] WARNING: no 'expert_projector' keys found in "
                  f"{config.warmup_connector_path}; nothing loaded.")
        else:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(f"[warmup] Loaded {len(filtered)} keys: {list(filtered.keys())}")
            if unexpected:
                print(f"[warmup] Unexpected keys (ignored): {unexpected}")

    # Run the training process
    run_training(config, train_dataset, eval_dataset, model, processor)