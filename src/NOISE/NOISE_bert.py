#!/usr/bin/env python3


from __future__ import annotations
from src.utils.seed import set_seed
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from src.utils import importer


#PROJECT_ROOT = Path(__file__).resolve().parent.parent
#sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# Model Creation (Full Model, Unfrozen)
# -----------------------------------------------------------------------------
def build_full_base_model(
        model_name: str, num_labels:int
) -> nn.Module:

    # Load raw BERT sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_name,
        num_labels=num_labels
    )

    # Calculate total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full Fine-Tuning active. Total trainable parameters: {total_params:,}")

    return model




# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    lora = importer._import_lora()
    args = lora.parse_args()
    set_seed(args.seed)

    device = lora.detect_device(args.device)

    # ------------------------------------------------------------------ #
    # Load data
    # ------------------------------------------------------------------ #
    print("Loading data...")

    train_dataloader = lora.create_dataloader(
        file_path=args.train_data_path,
        tokenizer_path="bert-base-uncased",
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True
    )
    print(train_dataloader.dataset.__len__())

    val_dataloader = lora.create_dataloader(
        file_path=args.val_data_path,
        tokenizer_path="bert-base-uncased",
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False
    )
    print(val_dataloader.dataset.__len__())

    test_dataloader = lora.create_dataloader(
        file_path=args.test_data_path,
        tokenizer_path="bert-base-uncased",
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=False
    )

    print(test_dataloader.dataset.__len__())

    # ------------------------------------------------------------------ #
    # Create base-model
    # ------------------------------------------------------------------ #

    model = build_full_base_model(args.model_name, args.num_labels)


    # ------------------------------------------------------------------ #
    # Run full-tuning
    # ------------------------------------------------------------------ #

    history= lora.train_any_model(
        model=model,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        noisy_or_clean_label=args.target_label_col,
        device=device,
        n_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm
    )
    print(history)



if __name__ == "__main__":
    main()
