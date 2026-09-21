"""
Main CLI Entry Point for the Image Captioning Pipeline.

Supports the following commands:
    python main.py train                        Train the model (preprocess → train)
    python main.py predict --image <path>       Generate a caption for an image

Examples:
    python main.py train
    python main.py predict --image ./data/flickr8k/Images/sample.jpg
    python main.py predict --image D:/photos/beach.jpg
"""

import argparse
import sys


def main():
    # ------------------------------------------------------------------
    # Top-level parser
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="image_captioning",
        description=(
            "Automated Image Captioning - CNN-LSTM Pipeline\n"
            "Combines a pretrained InceptionV3 CNN with an LSTM decoder\n"
            "to generate descriptive captions for travel photographs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ------------------------------------------------------------------
    # 'train' subcommand
    # ------------------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Preprocess data and train the captioning model",
        description=(
            "Runs the full training pipeline:\n"
            "  1. Preprocess captions (clean, tokenize, save tokenizer)\n"
            "  2. Extract CNN features from all images\n"
            "  3. Train the LSTM decoder model\n"
            "  4. Save the best model and plot loss curves"
        ),
    )

    # ------------------------------------------------------------------
    # 'predict' subcommand
    # ------------------------------------------------------------------
    predict_parser = subparsers.add_parser(
        "predict",
        help="Generate a caption for a given image",
        description=(
            "Load the trained model and generate a natural-language caption\n"
            "for the specified image using greedy decoding."
        ),
    )
    predict_parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file to caption (e.g., ./photo.jpg)",
    )

    # ------------------------------------------------------------------
    # Parse arguments
    # ------------------------------------------------------------------
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dispatch commands
    # ------------------------------------------------------------------
    if args.command == "train":
        try:
            from train import train
            train()
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] Training failed: {e}")
            raise

    elif args.command == "predict":
        try:
            from predict import predict
            predict(args.image)
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Prediction failed: {e}")
            raise


if __name__ == "__main__":
    main()
