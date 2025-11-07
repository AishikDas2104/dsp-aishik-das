import argparse
from build_model import build_model
from house_prices.inference import make_predictions
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="House Prices ML Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data", type=str, required=True, help="Path to training data CSV")

    
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input", type=str, required=True, help="Path to input CSV")
    predict_parser.add_argument("--output", type=str, required=True, help="Path to save predictions")

    args = parser.parse_args()

    if args.command == "train":
        build_model(args.data)
    elif args.command == "predict":
        df = pd.read_csv(args.input)
        preds = make_predictions(df)
        pd.DataFrame(preds, columns=["PredictedPrice"]).to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
